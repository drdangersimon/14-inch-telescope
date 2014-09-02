import os
import multiprocessing
import subprocess
import datetime
from reduice import gui_getdir

astrcmd = 'solve-field' #astrometry.net command
sexcmd = 'sex' #source extractor command

def solve(filePath):
    '''Runs the Astrometry.net solve-field command on a file path.
    filePath is a string.
    The output path for Astrometry.net is set to ./astrometry_output'''
    
    fileName = filePath[filePath.rfind('/')+1:] 
    astLog = open('astrometry_log.txt','a')
    srcExLog = open('sourceExtractor_log.txt', 'a')    
    astLog.write('LOG FOR: %s at %s\n\n'%(fileName,datetime.datetime.now()))
    srcExLog.write('LOG FOR: %s at %s\n\n'%(fileName,datetime.datetime.now()))
    bashCommand = "%s -v --no-background-subtraction --scale-units arcsecperpix --scale-low .6 --scale-high 2.88 --overwrite --dir astrometry_output --no-plots --use-sextractor '%s'"%(astrcmd, filePath[2:])        
    print 'Solving %s ...'%fileName   
    process = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    #Create log file
    astOut = process.stdout.read()
    sexOut = process.stderr.read()    
    astLog.write(astOut+'\n')
    astLog.close()
    srcExLog.write(sexOut+'\n')
    srcExLog.close()
    a = logHandle(astOut, fileName)
    b = logHandle(sexOut, fileName)
    print "Solved %s"%fileName    
    if a == True or b == True:
        print "There have been warnings or errors while processing %s, you might want to check the error log"%fileName
        


def logHandle(logString, fileName):
    '''Searches for error messages in a string
    logString is a string
    Outputs error messages to screen and to a file log.txt'''
    
    logList = logString.lower().split('\n')
    logFile = open('error_log.txt','a')
    errors = False       
    for line in logList:           
        if line.find('run_command_get_outputs command failed') !=-1:
            logFile.write('ERROR while processing %s\nCannot solve file, please check the file\n%s\n\n'%(fileName,line))
            #print 'ERROR while processing %s\nCannot solve file, please check the file\n%s\n\n'%(fileName,line)
            errors = True
        elif line.find('reason') != -1:
            logFile.write('%s\n\n\n'%line)
            #print '%s\n'%line
            errors = True
        elif line.find('failed') != -1:            
            logFile.write('ERROR while processing %s\n%s\n\n'%(fileName,line))
            #print 'ERROR while processing %s\n%s\n'%(fileName,line)
            errors = True
        elif line.find('warning') !=-1:
            logFile.write('WARNING while processing %s\n%s\n\n'%(fileName,line))        
            #3print 'WARNING while processing %s\n%s\n'%(fileName,line)   
            errors = True
        elif line.find('not found') !=-1:
            logFile.write('ERROR while processing %s\n%s\n\n'%(fileName,line))
            #print 'ERROR while processing %s\n%s\n'%(fileName,line)
            errors = True
    logFile.close()
    return errors
        

def check(path,filename,fileList):
        '''Checks a directory for a file.
        path is a os.walk generated tuple.
        filename is a string a full or partial name to search for.
        fileList is a list that the file paths will be added to.
        Yields a list of strings which are file paths.

        Example:
        >>a=('/home/usr/Desktop/folder',['empy_folder'],['some_document.txt','my_document.txt'])        
        >>b=[]  
        >>check(a,'.txt',b)
        ['/home/usr/Desktop/folder/some_document.txt', '/home/usr/Desktop/folder/my_document.txt']'''
        
        for i in path[2]:
                if i.find(filename) != -1:
                        fileList.append('./'+path[0]+'/'+i)                     
        return fileList


def search(top, filename):
        '''Find files in directory tree.
        top is a string, the top directory of the directory tree.
        filename is a string, a full or partial name to search for.
        Yields a list of file paths'''
        
        a = os.walk(top)
        files = []

        try:
                while True:
                        b = a.next()    
                        check(b, filename, files)

        except(StopIteration):          
                if len(files)==0:
                    print '####ERROR####\nNo files found, check the directory and file name you are searching for.'
                else:
                    print 'Located the following files:'
                    for fit in files:
                        fileName = fit[fit.rfind('/')+1:]                        
                        print fileName
                return files

def test():
    '''Tests whether the Source Extractor and Astrometry.net commands are found'''
    
    p = subprocess.Popen(sexcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.stderr.read()#reads error output
    if output.find('not found') != -1:
        print '####ERROR####\nsextractor may not be properly installed: %s'%output
        return -1
    p = subprocess.Popen(astrcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.stderr.read()    
    if output.find('not found') != -1:
        print '####ERROR####\nThe Astrometry.net software may not be properly installed: %s'%output
        return -1
    else:
        print 'Test passed'
        return 1        
        
def main(path=None, filename='.fit', cpus=multiprocessing.cpu_count()):
    '''Searches a directory for specified files and runs the astrometry.net solve-field on them.
    top is a string, the directory you wish to search (including subfolders).
    filename is a string, the full or partial file name you would like to run solve-field on.
    Example:
    >>main('calibrated_iamges', '.fit')'''
    
    if test() == -1:
        exit()

    if path is None:
        path = gui_getdir(title='Please Select raw fits Directory')
    #check if getting path was successfull
    if path is None:
        raise TypeError('Must choose path, with GUI or argument.')
    
    pathList = search(path, filename)
    #cpus = multiprocessing.cpu_count()
    m = multiprocessing.Pool(cpus)
    w = m.map_async(solve,pathList)
    w.wait()
    print "\nEnd of astrometry solving!\n"


if __name__ == "__main__":
    import sys
    #check in command line input
    if len(sys.argv) > 0:
        main(sys.argv[1])
    else:
        main()
