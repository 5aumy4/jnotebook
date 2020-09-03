arr = ["sex", "gender" , "haemoglobin", "hemoglobin", "wbc count", "neutrophil", "lymphocyte", "monocyte", "eosinophil"
      "basophil", "platelet count", "bilirubin(total)", "bilirubin(conjugated)","bilirubin(unconjugated)", "total protein",
      "albumin", "globulin", "alkaline phosphatase", "blood sugar(f)", "blood sugar(pp)", "urea", "creatinine","cholestrol"
      "hematocrit", "haematocrit", "a/g ratio", "albumin/globulin", "albumin/globumin ratio", "a/g", "ast", "alt","bilirubin,total", "bilirubin, conjugated"
       ,"bilirubin, unconjugated", "protein, total","rbc count", "mcv" , "mch" , "mchc" , "rdw" , "tlc" , "lipid profile",
      "sugar" ,"complete blood count" ,"segmented neutrophils" , "lymphocytes" , "monocytes", "eosinophils" , "basophils",]
#Import required packages 
import cv2 
import pytesseract 
  
# Mention the installed location of Tesseract-OCR in your system 
#pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
  
# Read image from which text needs to be extracted 
img = cv2.imread("C:/Users/Saumya Sah/Desktop/drlal1.png") 
  
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
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 900)) 
  
# Appplying dilation on the threshold image 
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
  
# Finding contours 
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE) 
  
# Creating a copy of image 
im2 = img.copy() 
  
# A text file is created and flushed 
#file = open("recognized.txt", "w+") 
#file.write("") 
#file.close() 
  
# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
for cnt in contours: 
     x, y, w, h = cv2.boundingRect(cnt) 
      
    # Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w] 
      
    # Open the file in append mode 
    file = open("recognized.txt", "a") 
      
    # Apply OCR on the cropped image 
    text = pytesseract.image_to_string(cropped) 
    #print(text)
    text = text.lower()
    lit = text.split()
    for i in range(len(lit)):
        if lit[i] in arr:
            try:
                print(lit[i]+" "+lit[i+1]+" "+ lit[i+2])
            except:
                print(lit[i])
        try:        
            word = (lit[i] + " "+ lit[i+1])
            if word in arr :
                try:
                    print(word +" "+ lit[i+2] + " "+lit[i+3])
                except:
                    print(word)
        except:
            continue
    # Appending the text into file 
    file.write(text) 
    file.write("\n") 
      
    # Close the file 
    file.close    
