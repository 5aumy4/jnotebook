import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, 'C:/Users/Saumya Sah')

#from arr_up import arr_up
arr=[' glutamyl transpeptidase,alt (sgpt), serum', ' random urine',
       '(alp)', '(alp), serum', '(alp, serum)', '(alt)', '(ast)',
       '(beta hydroxybutyrate),', '(clia)', '(eag)', '(esr)', '(etg)',
        '(ets)', '(fli)', '(hb/hgb)', '(hct)', '(hplc)', '(icpms)',
        '(indirect ise)', '(mch)', '(mchc)', '(mcv)',
        '(parathyroid hormone)', '(rbc)', '(rdw)', '(sbc)', '(tco2)',
        '(thiamin pyrophosphate)', '(tlc)protein, total', '(wbc)',
        '11- deoxycortisol', '17- hydroxycorticosteroids ( 17-ohcs)',
        '17-hydroxypregnenolone17- hydroxycorticosteroids ( 17-ohcs)',
        '17-hydroxyprogesterone', '17-hydroxyprogesterone(17-ohp)',
        '17-ketosteroids', '17-ohp + androstenedione : cortisol ratio',
        '17-ohp, basal', '25-hydroxy vitamin d total',
       '25-hydroxy vitamin d2', '25-hydroxy vitamin d3', '5-hiaa', 'a/g',
       'a/g ratio', 'a:g ratio', 'abo group',
       'absolute lymphocyte count, whole blood',
       'absolute neutrophil count, whole blood',
       'active b12, (holotranscobalamin)', 'afp', 'albumin',
       'albumin, serum', 'albumin/globulin', 'albumin/globulin ratio',
       'aldosterone', 'alkaline phosphatase', 'alp', 'alpha 1 globulin',
       'alpha 2 globulin', 'alpha-1-acid glycoprotein',
       'alpha-2 macroglobulin', 'alt', 'alt (sgpt)', 'aluminum, plasma',
       'amylase, serum', 'androstenedione', 'apo b / apo a1 ratio',
       'apolipoprotein (apo a1)', 'apolipoprotein (apo b)',
       'arsenic blood', 'asr (sgot)', 'ast', 'ast (sgot)',
       'ast (sgot), serum', 'base excess',
       'base excess-extracellular fluid ', 'basophils',
       'beta 2 glycoprotein i, iga, serum', 'beta 2 microglobulin, serum',
       'beta cell function (%b)', 'beta crosslaps (beta ctx), plasma',
       'beta globulin', 'bicarbonate', 'bicarbonate (hco3)',
       'bilirubin direct', 'bilirubin indirect', 'bilirubin total',
       'bilirubin(conjugated)', 'bilirubin(total)',
       'bilirubin(unconjugated)', 'bilirubin, conjugated',
       'bilirubin, total, serum', 'bilirubin, unconjugated',
       'bilirubin,total', 'blood group', 'blood sugar(f)',
       'blood sugar(pp)', 'blood urea', 'bun', 'bupropian',
       'c3 complement ,serum', 'cadmium, blood', 'calcium, serum',
       'calcium, total', 'cardio c-reactive protein (hscrp), serum',
       'ceruloplasmin, serum', 'chloride', 'chloride, serum',
       'cholesterol, total', 'cholesterol, total, serum', 'cholestrol','cholesterol',
       'cholestrolhematocrit', 'chromium, blood', 'clq, serum',
       'cobalt, blood', 'complete blood count', 'corticosterone',
       'cortisol', 'cortisone', 'creatinine', 'creatinine, serum',
       'cyanocobalamin, serum', 'cytomegalovirus, igg',
       'deoxycorticosterone', 'desipramine', 'dhea', 'digoxin, serum',
       'dpd gene mutation', 'e.s.r. (westergren)', 'electrolytes, serum',
       'eosinophilbasophil', 'eosinophils', 'esr', 'estriol, free',
       'ethosuximide', 'extracellular fluid', 'ferritin, serum',
       'flecainide', 'folate (folic acid), serum',
       'free / total psa ratio', 'fsh', 'gamma globulin',
       'gamma interferon, antigen tube', 'gamma interferon, nil tube',
       'gastrin', 'gastrin, basal', 'gender', 'gfr, estimated', 'ggtp',
       'globulin', 'glucose', 'glucose plasma, 2 hours',
       'glucose plasma, fasting', 'glucose, fasting (f), plasma',
       'glucose, pp', 'glucose, random (r), plasma', 'glycohaemoglobin',
       'glycohaemoglobin(alc)','glycohaemoglobin (alc)','glycosylated haemoglobin[hba1c]*',
       'haematocrit', 'haemoglobin', 'haemoglobin (cyanmeth.)', 'hb',
       'hb a2', 'hb adult', 'hb f', 'hb, blood', 'hba1c',
       'hba1c (glycosylated hemoglobin), blood', 'hcg', 'hdl cholesterol',
       'hdl cholesterol, direct,','hdl-cholesterol (d)', 'ldl-cholesterol (d)','hdl-cholesterol', 'ldl-cholesterol',
       'hdl cholestrolldl cholestrol, calculated', 'hematocrit',
       'hematocrit,', 'hemoglobin', 'homa ir index',
       'hyaluronic acid (ha)', 'hydroxybupropian', 'hydroxyprogesterone',
       'iga', 'igg', 'igg serum', 'igm', 'inhibin a',
       'insulin sensitivity (%s)', 'insulin, random, serum',
       'insulin, serum , fasting', 'international normalized ratio (inr)',
       'iron', 'ldl cholesterol', 'leptin', 'leptin, serum', 'lh',
       'lipase, serum', 'lipid profile', 'lymphocyte', 'lymphocytes',
       'manganese, blood', 'mch', 'mchc', 'mcv',
       'mean corpuscular hemoglobin (mch),', 'mean platelet volume',
       'methotrexate, serum', 'monocyte', 'monocytes', 'mpv',
       'mpv, whole blood', 'n-acetylprocainamide', 'neutrophil',
       'non-hdl cholesterol', 'nordiazepam', 'ogpt', 'osmolality',
       'others (not specific)', 'oxygen saturation capacity', 'pa+napa',
       'packed cell volume', 'pco2', 'pcv', 'peak 3', 'ph',
       'phenobarbitone,', 'phenol', 'phenytoin, serum', 'phosphorus',
       'phosphorus, serum', 'piiinp (procollagen type iii amino terminal)',
       'placental growth factor (plgf),serum', 'plasma',
       'plasma glucose (f)', 'plasma glucose (fasting)',
       'plasma glucose (pp)', 'plasma glucose (r)rdw', 'platelet count',
       'platelet count, whole blood', 'po2', 'potassium',
       'potassium, serum', 'procainamide', 'procalcitonin (pct), serum',
       'progesterone', 'progesterone, serum', 'prolactin, serum',
       'properdin factor b', 'protein c antigen', 'protein c, functional',
       'protein total, serum', 'protein, total', 'prothrombin ratio (pr)',
       'psa, free', 'psa, total', 'rbc count', 'rdw', 'rdw-cv', 'rdw-sd',
       'red blood cell count (rbc count)', 'wholered cell distribution width (rdw), whole',
       'reticulocyte count, whole blood', 'rh factor', 'rhd type',
       'rubella, igg', 'segmented neutrophils', 'serum creatinine',
       'serum hbsag', 'serum t3', 'serum t4',
       'serum tshtotal triiodothyronine(tt3)', 'serum urea', 'sex',
       'sgot(ast)', 'sgpt(alt)', 'sirolimus, whole', 'sodium',
       'sodium : osmolality ratio', 'sodium, serum', 'somatostatin',
       'ss-a/ro', 'ss-b/la', 'succinylacetone', 'sugar', 't3 auto ab',
       't3, total', 't3, total, serum', 't4, free; ft4', 't4, total',
       't4, total, serumthyroxine ab', 't4, totaltsh',
       'tacrolimus (fk-506), whole blood', 'testosterone total',
       'testosterone, basal', 'testosterone, free, serum',
       'testosterone, post', 'thallium, blood', 'thyrotropin (tsh)',
       'tlc', 'tlc (total leucocyte count), whole blood', 'topiramate',
       'total co2 (tco2)', 'total iron binding capacity', 'total protein',
       'total thyroxine(tt4)', 'total urine volume',
       'toxoplasma avidity, igg,', 'toxoplasma igg', 'toxoplasma igm',
       'toxoplasma, igg', 'toxoplasma, igm', 'transferrin saturation',
       'triglycerides', 'triglyceride','trypsin', 'tsh', 'tsh , ultrasensitive,',
       'tsh, serum', 'tsh, ultrasensitive', 'typhi dot, serum',
       'tyrosine', 'urea', 'urea, serum', 'uric acid', 'uric acid, serum',
       'urine r/e', 'vitamin a (retinol)',
       'vitamin b1 (thiamin pyrophosphate)', 'vitamin b12',
       'vitamin b12 binding cap', 'vitamin b12; cyanocobalamin, serum',
       'vitamin d, 1, 25 dihydroxy, serum',
       'vitamin d, 25 - hydroxy, serum', 'vitamin e (tocopherol),',
       'vitamin e; tocopherol, serum', 'vitamin k1', 'vldl cholesterol',
       'vldl cholestrol,calculated', 'vldl-cholesterol','vwf', 'wbc count']

arrnum = [176,46,47,48,49,50,51,52,53,54,56,57,55,44]
def appdata(string):
    for char in string:
        if ord(char) in arrnum:
            continue
        else:
            return False
    return True
import cv2 
import pytesseract 
import json
  
# Mention the installed location of Tesseract-OCR in your system 
#pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
  
# Read image from which text needs to be extracted 
img = cv2.imread("C:/Users/Saumya Sah/Desktop/bloodrepo22.jpeg") 
  
# Preprocessing the image starts 
  
# Convert the image to gray scale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
# Performing OTSU threshold 
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
  
# Specify structure shape and kernel size.  
# Kernel size increases or decreases the area  
# of the rectangle to be detected. 
# A smaller value like (10, 10) will detect  
# each word instead of a sentence. 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5000)) 
  
# Appplying dilation on the threshold image 
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
  
# Finding contours 
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE) 
  
# Creating a copy of image 
im2 = img.copy() 
  
# A text file is created and flushed 
file = open("recognized0.txt", "w+") 
file.write("") 
file.close() 
  
# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
values = dict()
for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt) 
      
    # Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w] 
      
    # Open the file in append mode 
    
      
    # Apply OCR on the cropped image 
    text = pytesseract.image_to_string(cropped) 
    #print(text)
    text = text.lower()
    lit = text.split()
    num = len(lit)
    for i in range(len(lit)):
        
            
        if lit[i] in arr:
                
            try:
                print(lit[i]+" "+lit[i+1]+" "+ lit[i+2])
                if appdata(lit[i+1]):
                    values[lit[i]]= lit[i+1]
                elif appdata(lit[i+2]):
                    values[lit[i]]= lit[i+2]
                #file = open("recognized0.txt", "a") 
                #file.write(lit[i]+" "+lit[i+1]+" "+ lit[i+2])
                #file.write("\n")
                #file.close
            except:
                print(lit[i])
        try:        
            word = (lit[i] + " "+ lit[i+1])
            if word in arr :
                try:
                    print(word +" "+ lit[i+2] + " "+lit[i+3])
                    if appdata(lit[i+2]):
                        values[word]= lit[i+2]
                    elif appdata(lit[i+3]):
                        values[word]= lit[i+3]
                    #file = open("recognized0.txt", "a") 
                    #file.write(word+" "+lit[i+2]+" "+ lit[i+3])
                    #file.write("\n")
                    #file.close
                except:
                    print(word) 
        except:
            continue
        try:        
            word = (lit[i] + " "+ lit[i+1]+" "+lit[i+2])
            if word in arr :
                try:
                    print(word +" "+ lit[i+3] + " "+lit[i+4])
                    if appdata(lit[i+3]):
                        values[word]= lit[i+3]
                    elif appdata(lit[i+4]):
                        values[word]= lit[i+4]
                    #file = open("recognized0.txt", "a") 
                    #file.write(word+" "+lit[i+3]+" "+ lit[i+4])
                    #file.write("\n")
                    #file.close
                except:
                    print(word)
        except:
            continue
        try:        
            word = (lit[i] + " "+ lit[i+1]+" "+lit[i+2]+" "+lit[i+3])
            if word in arr :
                try:
                    print(word +" "+ lit[i+4] + " "+lit[i+5])
                    if appdata(lit[i+4]) :
                        values[word]= lit[i+4]
                    elif appdata(lit[i+5]):
                        values[word]= lit[i+5]
                    #file = open("recognized0.txt", "a") 
                    #file.write(word+" "+lit[i+4]+" "+ lit[i+5])
                    #file.write("\n")
                    #file.close
                except:
                    print(word)
        except:
            continue
        try:        
            word = (lit[i] + " "+ lit[i+1]+" "+lit[i+2]+" "+lit[i+3]+" "+lit[i+4])
            if word in arr :
                try:
                    print(word +" "+ lit[i+5] + " "+lit[i+6])
                    if appdata(lit[i+5]):
                        values[word]= lit[i+5]
                    elif appdata(lit[i+6]):
                        values[word]= lit[i+6]
                    #file = open("recognized0.txt", "a") 
                    #file.write(word+" "+lit[i+5]+" "+ lit[i+6])
                    #file.write("\n")
                    #file.close
                except:
                    print(word)
        except:
            continue
        try:        
            word = (lit[i] + " "+ lit[i+1]+" "+lit[i+2]+" "+lit[i+3]+" "+lit[i+4]+" "+lit[i+5])
            if word in arr :
                try:
                    print(word +" "+ lit[i+6] + " "+lit[i+7])
                    if appdata(lit[i+6]):
                        values[word]= lit[i+6]
                    elif appdata(lit[i+7]):
                        values[word]= lit[i+7]
                    #file = open("recognized0.txt", "a") 
                    #file.write(word+" "+lit[i+6]+" "+ lit[i+7])
                    #file.write("\n")
                    #file.close
                except:
                    print(word)
        except:
            continue
values
out_file = open("test.json", "w") 
json.dump(values, out_file, indent = 4, sort_keys = False) 
out_file.close() 
