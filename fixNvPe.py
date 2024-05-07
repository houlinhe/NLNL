# Simple script to disable ASLR and make .nv_fatb sections read-only
# Requires: pefile  ( python -m pip install pefile )
# Usage:  fixNvPe.py --input path/to/*.dll

import argparse
import pefile
import glob
import os
import shutil
# python fixNvPe.py --input=C:\Users\User\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\lib\*.dll
# https://stackoverflow.com/questions/64837376/how-to-efficiently-run-multiple-pytorch-processes-models-at-once-traceback
def main(args):
    failures = []
    for file in glob.glob( args.input, recursive=args.recursive ):
        print(f"\n---\nChecking {file}...")
        pe = pefile.PE(file, fast_load=True)
        nvbSect = [ section for section in pe.sections if section.Name.decode().startswith(".nv_fatb")]
        if len(nvbSect) == 1:
            sect = nvbSect[0]
            size = sect.Misc_VirtualSize
            aslr = pe.OPTIONAL_HEADER.IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
            writable = 0 != ( sect.Characteristics & pefile.SECTION_CHARACTERISTICS['IMAGE_SCN_MEM_WRITE'] )
            print(f"Found NV FatBin! Size: {size/1024/1024:0.2f}MB  ASLR: {aslr}  Writable: {writable}")
            if (writable or aslr) and size > 0:
                print("- Modifying DLL")
                if args.backup:
                    bakFile = f"{file}_bak"
                    print(f"- Backing up [{file}] -> [{bakFile}]")
                    if os.path.exists( bakFile ):
                        print( f"- Warning: Backup file already exists ({bakFile}), not modifying file! Delete the 'bak' to allow modification")
                        failures.append( file )
                        continue
                    try:
                        shutil.copy2( file, bakFile)
                    except Exception as e:
                        print( f"- Failed to create backup! [{str(e)}], not modifying file!")
                        failures.append( file )
                        continue
                # Disable ASLR for DLL, and disable writing for section
                pe.OPTIONAL_HEADER.DllCharacteristics &= ~pefile.DLL_CHARACTERISTICS['IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE']
                sect.Characteristics = sect.Characteristics & ~pefile.SECTION_CHARACTERISTICS['IMAGE_SCN_MEM_WRITE']
                try:
                    newFile = f"{file}_mod"
                    print( f"- Writing modified DLL to [{newFile}]")
                    pe.write( newFile )
                    pe.close()
                    print( f"- Moving modified DLL to [{file}]")
                    os.remove( file )
                    shutil.move( newFile, file )
                except Exception as e:
                    print( f"- Failed to write modified DLL! [{str(e)}]")
                    failures.append( file )
                    continue

    print("\n\nDone!")
    if len(failures) > 0:
        print("***WARNING**** These files needed modification but failed: ")
        for failure in failures:
            print( f" - {failure}")