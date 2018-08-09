
import numpy as np
from astropy.io import registry
from astropy.table import Table, Column
from .utils import (species_to_element, species_to_elems_isotopes_ion)

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s"))
logger.addHandler(handler)

import os
from hashlib import md5


class LineList(Table):
    full_colnames = ['wavelength','species','expot','loggf','damp_vdw','dissoc_E','comments',
                     'numelems','elem1','isotope1','elem2','isotope2','ion',
                     'E_hi','lande_hi','lande_lo','damp_stark','damp_rad','references','element']
    full_dtypes = [np.float,np.float,np.float,np.float,np.float,np.float,str,
                   np.int,str,np.int,str,np.int,np.int,
                   np.float,np.float,np.float,np.float,np.float,str,str]
    moog_colnames = ['wavelength','species','expot','loggf','damp_vdw','dissoc_E','comments',
                     'numelems','elem1','isotope1','elem2','isotope2','ion',
                     'references','element']
    moog_dtypes = [np.float,np.float,np.float,np.float,np.float,np.float,str,
                   np.int,str,np.int,str,np.int,np.int,
                   str,str]

    def __init__(self,*args,**kwargs):
        # Pull out some default kwargs
        if 'verbose' in kwargs: 
            self.verbose = kwargs.pop('verbose')
        else:
            self.verbose = False
        if 'moog_columns' in kwargs: 
            self.moog_columns = kwargs.pop('moog_columns')
        else:
            self.moog_columns = False
        if 'default_thresh' in kwargs:
            self.default_thresh = kwargs.pop('default_thresh')
        else:
            self.default_thresh = 0.1
        if 'default_loggf_thresh' in kwargs:
            self.default_loggf_thresh = kwargs.pop('default_loggf_thresh')
        else:
            self.default_loggf_thresh = 0.01
        if 'default_expot_thresh' in kwargs:
            self.default_expot_thresh = kwargs.pop('default_expot_thresh')
        else:
            self.default_expot_thresh = 0.01
        if "check_for_duplicates" in kwargs:
            # If you check for duplicates, you do not have duplicates
            # (because a ValueError is thrown otherwise)
            self.has_duplicates = ~kwargs.pop("check_for_duplicates")
        else:
            # By default, do NOT check for duplicates
            self.has_duplicates = True

        super(LineList, self).__init__(*args,**kwargs)

        if 'hash' not in self.columns and len(self) > 0:
            # When sorting, it creates a LineList with just a column subset
            try: 
                self.validate_colnames(True)
            except IOError:
                pass
            else:
                hashes = [self.hash(line) for line in self]
                self.add_column(Column(hashes,name='hash'))

        if 'hash' in self.columns and (not self.has_duplicates):
            self.check_for_duplicates()
        #self.validate_colnames(False)

    def check_for_duplicates(self):
        """
        Check for exactly duplicated lines. This has to fail because hashes
        are assumed to be unique in a LineList.
        Exactly duplicated lines may occur for real reasons, e.g. if there is
        insufficient precision to distinguish two HFS lines in the line list.
        In these cases, it may be okay to combine the two lines into one 
        total line with a combined loggf.
        """
        if len(self) != len(np.unique(self['hash'])):
            error_msg = \
                "This LineList contains lines with identical hashes.\n" \
                "The problem is most likely due to completely identical lines\n" \
                "(e.g., because of insufficient precision in HFS).\n" \
                "If that is the case, it may be reasonable to combine the\n" \
                "loggf for the two lines into a single line.\n" \
                "We now print the duplicated lines:\n"
            fmt = "{:.3f} {:.3f} {:.3f} {:5} {}\n"
            total_duplicates = 0
            for i,hash in enumerate(self['hash']):
                N = np.sum(self['hash']==hash)
                if N > 1: 
                    line = self[i]
                    total_duplicates += 1
                    error_msg += fmt.format(line['wavelength'],line['expot'],line['loggf'],line['element'],line['hash'])
            raise ValueError(error_msg)
        self.has_duplicates = False
        return None

    def validate_colnames(self,error=False):
        """
        error: if True, raise error when validating.
            This is False by default because many astropy.table operations
            create empty or small tables.
        """
        ## This is included b'c table.vstack() creates an empty table
        if len(self.columns)==0: return False

        if self.moog_columns:
            colnames = self.moog_colnames
        else:
            colnames = self.full_colnames
        badcols = []
        for col in colnames:
            if col not in self.columns: badcols.append(col)
        
        error_msg = "Missing columns: {}".format(badcols)
        if len(badcols) == 0: return True
        if error:
            raise IOError(error_msg)
        else:
            print(error_msg)
        return False


    @property
    def unique_elements(self):
        """ Return the unique elements that are within this line list. """

        elements = list(self["elem1"]) + list(self["elem2"])
        return list(set(elements).difference([""]))



    @staticmethod
    def hash(line):
        s = "{:.3f}_{:.3f}_{:.3f}_{}_{}_{}_{}_{}".format(line['wavelength'],line['expot'],line['loggf'],
                                                         line['elem1'],line['elem2'],line['ion'],line['isotope1'],line['isotope2'])
        #return md5.new(s).hexdigest()
        m = md5()
        m.update(str.encode(s))
        return m.hexdigest()



    @classmethod
    def read(cls,filename,*args,**kwargs):
        """
        filename: name of the file
        To use the default Table reader, must specify 'format' keyword.
        Otherwise, tries to read moog and then GES fits
        """
        
        if 'format' in kwargs: 
            return cls(super(LineList, cls).read(*((filename,)+args), **kwargs))

        if not os.path.exists(filename):
            raise IOError("No such file or directory: {}".format(filename))
        for reader in (cls.read_moog, ):
            try:
                return reader(filename,**kwargs)
            except (IOError, KeyError, UnicodeDecodeError):
                # KeyError: Issue #87
                # UnicodeDecodeError: read_moog fails this way for fits
                pass
        raise IOError("Cannot identify linelist format (specify format if possible)")

    @classmethod
    def read_moog(cls,filename,moog_columns=False,**kwargs):
        if moog_columns:
            colnames = cls.moog_colnames
            dtypes = cls.moog_dtypes
        else:
            colnames = cls.full_colnames
            dtypes = cls.full_dtypes
    
        with open(filename) as f:
            lines = f.readlines()
        N = len(lines)
        wl = np.zeros(N)
        species = np.zeros(N)
        EP = np.zeros(N)
        loggf = np.zeros(N)
        ew = np.zeros(N) * np.nan
        damping = np.zeros(N) * np.nan
        dissoc = np.zeros(N) * np.nan
        comments = ['' for i in range(N)]
        # wl, transition, EP, loggf, VDW damping C6, dissociation D0, EW, comments
        has_header_line = False
        for i,line in enumerate(lines):
            s = line.split()
            try:
                _wl,_species,_EP,_loggf = map(float,s[:4])
            except:
                if i==0:
                    has_header_line = True
                    continue
                else:
                    raise IOError("Invalid line: {}".format(line))
            if len(s) > 4:
                try: damping[i] = float(line[40:50])
                except: pass
                
                try: dissoc[i] = float(line[50:60])
                except: pass
                
                try: 
                    _ew = float(line[60:70])
                    # It seems some linelists have -1 in the EW location as a placeholder
                    if _ew <= 0: 
                        raise ValueError("EW <= 0: {}".format(_ew))
                    ew[i] = _ew
                    comments[i] = line[70:].strip()
                except:
                    comments[i] = line[60:].strip()
            wl[i] = _wl; species[i] = _species; EP[i] = _EP; loggf[i] = _loggf
        if has_header_line:
            wl = wl[1:]; species = species[1:]; EP = EP[1:]; loggf = loggf[1:]
            damping = damping[1:]; dissoc = dissoc[1:]; comments = comments[1:]
            ew = ew[1:]
        
        # check if gf by assuming there is at least one line with loggf < 0
        if np.all(loggf >= 0): 
            loggf = np.log10(loggf)
            # TODO this is the MOOG default, but it may not be a good idea...
            print("Warning: no lines with loggf < 0 in {}, assuming input is gf".format(filename))
        
        # TODO
        # Cite the filename as the reference for now
        refs = [filename for x in wl]
    
        # Species to element
        spec2element = {}
        spec2elem1= {}
        spec2elem2= {}
        spec2iso1 = {}
        spec2iso2 = {}
        spec2ion  = {}
        for this_species in np.unique(species):
            spec2element[this_species] = species_to_element(this_species)
            _e1, _e2, _i1, _i2, _ion = species_to_elems_isotopes_ion(this_species)
            spec2elem1[this_species] = _e1
            spec2elem2[this_species] = _e2
            spec2iso1[this_species] = _i1
            spec2iso2[this_species] = _i2
            spec2ion[this_species] = _ion
        numelems = np.array([2 if x >= 100 else 1 for x in species])
        elements = [spec2element[this_species] for this_species in species]
        elem1 = [spec2elem1[this_species] for this_species in species]
        elem2 = [spec2elem2[this_species] for this_species in species]
        isotope1 = [spec2iso1[this_species] for this_species in species]
        isotope2 = [spec2iso2[this_species] for this_species in species]
        ion  = [spec2ion[this_species] for this_species in species]

        # Fill required non-MOOG fields with nan
        if moog_columns:
            data = [wl,species,EP,loggf,damping,dissoc,comments,
                    numelems,elem1,isotope1,elem2,isotope2,ion,
                    refs,elements]
        else:
            nans = np.zeros_like(wl)*np.nan
            E_hi = EP + 12398.42/wl #hc = 12398.42 eV AA
            data = [wl,species,EP,loggf,damping,dissoc,comments,
                    numelems,elem1,isotope1,elem2,isotope2,ion,
                    E_hi,nans,nans,nans,nans,refs,elements]
        # add EW if needed
        #if not np.all(np.isnan(ew)):
        #    print("Read {} EWs out of {} lines".format(np.sum(~np.isnan(ew)),len(ew)))
        colnames = colnames + ['equivalent_width']
        dtypes = dtypes + [np.float]
        data = data + [ew]
        
        return cls(Table(data,names=colnames,dtype=dtypes),moog_columns=moog_columns,**kwargs)


    def write_moog(self,filename):
        fmt = "{:10.3f}{:10.5f}{:10.3f}{:10.3f}{}{}{} {}"
        space = " "*10
        with open(filename,'w') as f:
            f.write("\n")
            for line in self:
                C6 = space if np.ma.is_masked(line['damp_vdw']) or np.isnan(line['damp_vdw']) else "{:10.3f}".format(line['damp_vdw'])
                D0 = space if np.ma.is_masked(line['dissoc_E']) or np.isnan(line['dissoc_E']) else "{:10.3}".format(line['dissoc_E'])
                #comments = '' if np.ma.is_masked(line['comments']) else line['comments']
                comments = ""
                if 'equivalent_width' in line.colnames:
                    EW = space if np.ma.is_masked(line['equivalent_width']) or np.isnan(line['equivalent_width']) else "{:10.3f}".format(line['equivalent_width'])
                else:
                    EW = space
                f.write(fmt.format(line['wavelength'],line['species'],line['expot'],line['loggf'],C6,D0,EW,comments)+"\n")


    def write_latex(self,filename,sortby=['species','wavelength'],
                    write_cols = ['wavelength','element','expot','loggf']):
        new_table = self.copy()
        new_table.sort(sortby)
        new_table = new_table[write_cols]
        new_table.write(filename,format='aastex')

## Add to astropy.io registry
def _moog_identifier(*args, **kwargs):
    return isinstance(args[0], (str, )) and args[0].lower().endswith(".moog")
registry.register_writer("moog", LineList, LineList.write_moog)
registry.register_reader("moog", LineList, LineList.read_moog)
registry.register_identifier("moog", LineList, _moog_identifier)


