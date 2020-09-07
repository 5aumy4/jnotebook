# Import required packages
import cv2
import pytesseract
import json

# Mention the installed location of Tesseract-OCR in your system
#pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'

# Read image from which text needs to be extracted
img = cv2.imread("C:/Users/Saumya Sah/Desktop/bloodrepo5.jpeg")

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
                values[lit[i]]= lit[i+1]
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
                    values[word]= lit[i+2]
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
                    values[word]= lit[i+3]
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
                    values[word]= lit[i+3]
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
                    values[word]= lit[i+3]
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
                    values[word]= lit[i+3]
                    #file = open("recognized0.txt", "a")
                    #file.write(word+" "+lit[i+6]+" "+ lit[i+7])
                    #file.write("\n")
                    #file.close
                except:
                    print(word)
        except:
            continue
values
out_file = open("test1.json", "w")
json.dump(values, out_file, indent = 4, sort_keys = False)
out_file.close()
    # Appending the text into file
    #file.write(text)
    #file.write("\n")

    # Close the file
    #file.close 
