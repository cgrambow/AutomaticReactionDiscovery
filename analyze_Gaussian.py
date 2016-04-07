# encoding: utf-8

#! c:/python27
import sys
import glob
import string
import os
import math
import time
import re
import operator 
import subprocess


def gaussian_workout ( file ):
 file_output = []

 crg_index = -100
 inp = open(file,'r')
 for i, line in enumerate(inp):
   if 'Charge' in line and 'Multiplicity' in line:
     crg_index = i
 inp.close()

 if crg_index != -100:
  inp = open(file,'r')
  data0 = inp.readlines()
  inp.close()
  file_output.append(data0[crg_index])

 acrdn_index = -100
 inp = open(file,'r')
 for i, line in enumerate(inp):
   if "Standard orientation: " in line :
     acrdn_index = i
 inp.close()


 if acrdn_index != -100:
  inp = open(file,'r')
  data = inp.readlines()
  inp.close()
  j = acrdn_index
  while j > acrdn_index-1 and j < acrdn_index+6 : 
     file_output.append(data[j])
     j +=1
  while '--' not in data[j] :
     file_output.append(data[j])
     j +=1
  file_output.append(data[j])
  file_output.append(data[j+1])
  file_output.append(data[j+2])


 ptcglstM_index = -100
 inp = open(file,'r')
 for i, line in enumerate(inp) :
   if "Mulliken" in line and "charges:" in line :
     ptcglstM_index = i
 inp.close()
 
 if ptcglstM_index != -100 :
  inp = open(file,'r')
  data0 = inp.readlines()
  inp.close()
  j = ptcglstM_index
  while j > ptcglstM_index-1 and 'Sum of Mulliken' not in data0[j] :
      file_output.append(data0[j])
      j +=1
  file_output.append(data0[j])
  file_output.append(data0[j+1])
  file_output.append(data0[j+2])



 ptcglst_index= -100
 inp = open(file,'r')
 for i, line in enumerate(inp):
   if "Summary of Natural Population Analysis" in line:
     ptcglst_index = i
 inp.close()

 if ptcglst_index != -100 :
  inp = open(file,'r')
  data0 = inp.readlines()
  inp.close()
  j = ptcglst_index
  while j > ptcglst_index-1 and j < ptcglst_index+7 :
      file_output.append(data0[j])
      j +=1
  while '--' not in data0[j] :
      file_output.append(data0[j])
      j +=1
  file_output.append(data0[j])
  file_output.append(data0[j+1])
#  file_output.append(data0[j+2])

 scf_index = -100
 inp = open(file,'r')
 for i, line in enumerate(inp):
   if 'SCF Done:' in line:
     scf_index = i
 inp.close()

 if scf_index != -100:
  inp = open(file,'r')
  data0 = inp.readlines()
  inp.close()
  file_output.append(data0[scf_index])

 lastlines = [] 
 inp = open(file,'r')
 lastlines = tail(inp, 10, offset=None)
 for line in lastlines : 
  file_output.append(str(line)+'\n')
 
 inp = open(file,'wb')
 for item in file_output:
   inp.write(str(item))

 return "finished"


def gaussian_workout2 ( file, multiplicity, charge, cart, ptcglstM, ptcglst, energy, nbolst, nbolst_val, lastline ):

 anrlst = [' 1',' 5',' 6',' 7',' 8',' 9',         # atomic number list
          '14','15','16','17',' 3']
 alst = ['H','B','C','N','O','F',           # list of atoms corresponding
        'Si','P','S','Cl','Li']                          # to atomic numbers in anrlst

 dlst = [[0.9,1.2,1.2,1.3,1.5,1.4,1.6,1.43,1.5,1.6],   # interatomic distance list
        [1.2,1.8,1.72,1.7,1.6,1.5,2.1,2.04,1.6,1.7],   # threshold 10% : Si-H; 
        [1.2,1.72,1.66,1.56,1.52,1.55,2.1,1.95,1.94,1.85],   # 
        [1.3,1.8,1.56,1.7,1.45,1.4,2.04,1.8,1.8,1.9],   
        [1.5,1.7,1.52,1.45,1.8,1.4,2.2,1.8,1.9,2.0],   # threshold +0.07 : N-O;
        [1.4,1.5,1.55,1.4,1.4,1.9,2.2,1.9,2.0,2.1],   # threshold 10% : N-F;
        [1.6,2.1,2.1,2.04,2.2,2.2,2.3,2.3,2.25,2.4],   # threshold 10% : C-O;
        [1.43,2.04,1.95,1.8,1.8,1.9,2.3,2.3,2.1,2.3],
        [1.5,1.6,1.94,1.8,1.9,2.0,2.25,2.1,2.1,2.3],
        [1.5,1.6,1.85,1.8,1.9,2.0,2.4,2.3,2.2,2.3]]
 spc = "   "
 ss1 = len(anrlst)
 data = []
 data0 = []

 charge = "N/A"
 multiplicity = "N/A"

 inp = open(file,'r')
 for line in inp:
  if "Multiplicity" in line : 
    charge = int(line[10:12]) 
    multiplicity = line[28]
 inp.close()


 cart = []
 inp = open(file,'r')
 check01 = 0
 check1 = 0

 for line in inp.readlines():
        if "Standard orientation:" in line :
            check01 = 1
            check1 = 0
            cart = []
            continue
        elif check01 == 1 :
            anr = line[16:18]
            if anr == "--" :
                check1 +=1
                continue
            elif check1 == 3 :
                check01 = 0
                continue
            elif check1 == 2 :
                i = 0
                while i < ss1 :
                    if str(anr) == str(anrlst[i]) :
                        cart.extend([alst[i],line[36:46],
                                      line[48:58],line[60:-1]])
                        i +=1
                        continue
                    i+=1
                    continue
                continue
        else :
            continue
 inp.close()


 ptcglstM = []

 ptcglstM_index=-1000
 inp = open(file,'r')
 for i, line in enumerate(inp):
  if "Mulliken" in line and "charges:" in line:
    ptcglstM_index = i
 inp.close()
 if ptcglstM_index !=-1000:
  inp = open(file,'r')
  data0 = inp.readlines()
  j = ptcglstM_index+2
  while j > ptcglstM_index+1 and 'Sum of Mulliken' not in data0[j]:
           line = data0[j]
           temp = str(line[8:10])
           temp = temp.replace(" ","") 
           #SOSONEW if temp == "H" : 
           ptcglstM.extend([line[4:6],line[12:-1]])
           j +=1
  inp.close()

 ptcglst = []

 ptcglst_index=-1000
 inp = open(file,'r')
 for i, line in enumerate(inp):
  if "Summary of Natural Population Analysis" in line:
    ptcglst_index = i
 inp.close()
 if ptcglst_index != -1000:
  inp = open(file,'r')
  data0 = inp.readlines()
  j = ptcglst_index+6
  while j > ptcglst_index+5 and '==' not in data0[j]:
           line = data0[j]
           temp = str(line[6:8])
           temp = temp.replace(" ","")
           #SOSONEW if temp == "H" :   
           ptcglst.extend([int(line[10:12]),float(line[13:22])])
           j +=1
  inp.close()

 inp = open(file,'r')
 energy = "N/A"
 for line1 in inp.readlines():
                        if "SCF Done: " in line1 :
                            t = string.find(line1,".")
                            p = string.find(line1,"=")
                            energy = float(line1[(p+1):(t+10)])
                            continue
                        else :
                            continue
 inp.close()


 with open(file) as myfile:
   data = myfile.readlines() 
   lastline_index = len(data)-3 
   if 'Error' in data[lastline_index]: 
      lastline = data[lastline_index]

   lastline_index = len(data)-1 
   if 'Normal' in data[lastline_index]: 
      lastline = data[lastline_index]
 myfile.close() 
 
 if lastline == "N/A" :
       print 'ERROR while reading Gaussian output'

 nbolst = []
 nbolst_val = []
 myfile2 = open(file,'r')
 checkn2 = 0
 atmlstnuc = ['C','N','O','S','P','F','Cl','B','Si', 'Li']
 for line2 in myfile2.readlines(): # rabota s anionom
            if checkn2 == 0 :
                if " Summary of Natural Population Analysis:" in line2 : 
                    checkn2 +=1
                    continue
                else :
                    continue
            elif checkn2 == 2:
                atmchk = str(line2[5:7])
                atmchk = atmchk.replace(" ","")
                if atmchk in atmlstnuc :
                    atmnuc = atmchk
                    atmnumb = int(line2[10:12])
                    atmchrg = float(line2[15:23])
                    atmmol = atmnuc + str(atmnumb)
                    nbolst.append(atmmol)
                    nbolst_val.append(atmchrg)
                elif "==============================" in line2 :
                    break
                else :
                    continue
                continue
            elif checkn2 == 1:
                checkn2 +=1
                continue
 myfile2.close()

 return (multiplicity, charge, cart, ptcglstM, ptcglst, energy, nbolst, nbolst_val, lastline)




def gaussian_workout2_03 ( file, multiplicity, charge, cart, ptcglstM, ptcglst, energy, nbolst, nbolst_val, lastline ):

 anrlst = [' 1',' 5',' 6',' 7',' 8',' 9',         # atomic number list
          '14','15','16','17',' 3']
 alst = ['H','B','C','N','O','F',           # list of atoms corresponding
        'Si','P','S','Cl','Li']                          # to atomic numbers in anrlst

 dlst = [[0.9,1.2,1.2,1.3,1.5,1.4,1.6,1.43,1.5,1.6],   # interatomic distance list
        [1.2,1.8,1.72,1.7,1.6,1.5,2.1,2.04,1.6,1.7],   # threshold 10% : Si-H; 
        [1.2,1.72,1.66,1.56,1.52,1.55,2.1,1.95,1.94,1.85],   # 
        [1.3,1.8,1.56,1.7,1.45,1.4,2.04,1.8,1.8,1.9],   
        [1.5,1.7,1.52,1.45,1.8,1.4,2.2,1.8,1.9,2.0],   # threshold +0.07 : N-O;
        [1.4,1.5,1.55,1.4,1.4,1.9,2.2,1.9,2.0,2.1],   # threshold 10% : N-F;
        [1.6,2.1,2.1,2.04,2.2,2.2,2.3,2.3,2.25,2.4],   # threshold 10% : C-O;
        [1.43,2.04,1.95,1.8,1.8,1.9,2.3,2.3,2.1,2.3],
        [1.5,1.6,1.94,1.8,1.9,2.0,2.25,2.1,2.1,2.3],
        [1.5,1.6,1.85,1.8,1.9,2.0,2.4,2.3,2.2,2.3]]
 spc = "   "
 ss1 = len(anrlst)
 data = []
 data0 = []

 charge = "N/A"
 multiplicity = "N/A"

 inp = open(file,'r')
 for line in inp:
  if "Multiplicity" in line : 
    charge = int(line[10:12]) 
    multiplicity = line[28]
 inp.close()


 cart = []
 inp = open(file,'r')
 check01 = 0
 check1 = 0

 for line in inp.readlines():
        if "Standard orientation:" in line :
            check01 = 1
            check1 = 0
            cart = []
            continue
        elif check01 == 1 :
            anr = line[14:16]
            if anr == "--" :
                check1 +=1
                continue
            elif check1 == 3 :
                check01 = 0
                continue
            elif check1 == 2 :
                i = 0
                while i < ss1 :
                    if str(anr) == str(anrlst[i]) :
                        cart.extend([alst[i],line[36:46],
                                      line[48:58],line[60:-1]])
                        i +=1
                        continue
                    i+=1
                    continue
                continue
        else :
            continue
 inp.close()


 ptcglstM = []

 ptcglstM_index=-1000
 inp = open(file,'r')
 for i, line in enumerate(inp):
  if "Mulliken" in line and "charges:" in line:
    ptcglstM_index = i
 inp.close()
 if ptcglstM_index !=-1000:
  inp = open(file,'r')
  data0 = inp.readlines()
  j = ptcglstM_index+2
  while j > ptcglstM_index+1 and 'Sum of Mulliken' not in data0[j]:
           line = data0[j]
           temp = str(line[8:10])
           temp = temp.replace(" ","")
           #SOSONEW if temp == "H" : 
           ptcglstM.extend([line[4:6],line[12:-1]])
           j +=1
  inp.close()

 ptcglst = []

 ptcglst_index=-1000
 inp = open(file,'r')
 for i, line in enumerate(inp):
  if "Summary of Natural Population Analysis" in line:
    ptcglst_index = i
 inp.close()
 if ptcglst_index !=-1000:
  inp = open(file,'r')
  data0 = inp.readlines()
  j = ptcglst_index+6
  while j > ptcglst_index+5 and '==' not in data0[j]:
           line = data0[j]
           temp = str(line[6:8])
           temp = temp.replace(" ","")
           #SOSONEW if temp == "H" :   
           ptcglst.extend([int(line[10:12]),float(line[13:22])])
           j +=1
  inp.close()

 inp = open(file,'r')
 energy = "N/A"
 for line1 in inp.readlines():
                        if "SCF Done: " in line1 :
                            t = string.find(line1,".")
                            p = string.find(line1,"=")
                            energy = float(line1[(p+1):(t+10)])
                            continue
                        else :
                            continue
 inp.close()


 with open(file) as myfile:
   data = myfile.readlines() 
   lastline_index = len(data)-3 
   if 'Error' in data[lastline_index]: 
      lastline = data[lastline_index]
#      myfile.close() 

   lastline_index = len(data)-1 
   if 'Normal' in data[lastline_index]: 
      lastline = data[lastline_index]
   myfile.close() 
 
   if lastline == "N/A" :
       print 'ERROR while reading Gaussian output'

 nbolst = []
 nbolst_val = []
 myfile2 = open(file,'r')
 checkn2 = 0
 atmlstnuc = ['C','N','O','S','P','F','Cl','B','Si', 'Li']
 for line2 in myfile2.readlines(): # rabota s anionom
            if checkn2 == 0 :
                if " Summary of Natural Population Analysis:" in line2 : 
                    checkn2 +=1
                    continue
                else :
                    continue
            elif checkn2 == 2:
                atmchk = str(line2[5:7])
                atmchk = atmchk.replace(" ","") 
                if atmchk in atmlstnuc :
                    atmnuc = atmchk 
                    atmnumb = int(line2[10:12])
                    atmchrg = float(line2[15:23])
                    atmmol = atmnuc + str(atmnumb)
                    nbolst.append(atmmol)
                    nbolst_val.append(atmchrg)
                elif "==============================" in line2 :
                    break
                else :
                    continue
                continue
            elif checkn2 == 1:
                checkn2 +=1
                continue
 myfile2.close()

 return (multiplicity, charge, cart, ptcglstM, ptcglst, energy, nbolst, nbolst_val, lastline)
