import cv2
import pytesseract
import matplotlib.pyplot as plt 
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
from skimage import measure
import imutils
global yoyo
yoyo=0

global check
check=1
def sort_cont(character_contours):
    """
    To sort contours from left to right
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes),
                                                      key=lambda b: b[1][i], reverse=False))
    return character_contours


def segment_chars(plate_img, fixed_width):
    """
    extract Value channel from the HSV format of image and apply adaptive thresholding
    to reveal the characters on the license plate
    """
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    T = threshold_local(V, 29, offset=15, method='gaussian')

    thresh = (V > T).astype('uint8') * 255

    thresh = cv2.bitwise_not(thresh)

    # resize the license plate region to a canoncial size
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # perform a connected components analysis and initialize the mask to store the locations
    # of the character candidates
    labels = measure.label(thresh, background=0)
    
    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    # loop over the unique components
    characters = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255

        cnts,heir = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]


        # ensure at least one contour was found in the mask
        if len(cnts) > 0:
            
            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solodity, and height ration for the component
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)

                cv2.drawContours(charCandidates, [hull], -1, 255, -1)
                print(1)

    contours, hier = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4  # value to be added to each dimension of the character
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if y > addPixel:
                y = y - addPixel
            else:
                y = 0
            if x > addPixel:
                x = x - addPixel
            else:
                x = 0
            temp = bgr_thresh[y:y + h + (addPixel * 2), x:x + w + (addPixel * 2)]

            characters.append(temp)
        return characters
    else:
        return None



class PlateFinder:
    def __init__(self):
        self.min_area = 4500  # minimum area of the plate
        self.max_area = 30000  # maximum area of the plate

        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)  # old window was (5,5)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)  # convert to gray
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)  # sobelX to get the vertical edges
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        element = self.element_structure
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_contours(self, after_preprocess):
        contours,hierachy = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours,hierachy= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)  # index of the largest contour in the area array

            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            rotatedPlate = plate
            if not self.ratioCheck(max_cntArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
                return plate, False, None
            return rotatedPlate, True, [x, y, w, h]
        else:
            return plate, False, None



    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
                if (characters_on_plate is not None and len(characters_on_plate) == 8):
                    print(1)
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    after_check_plate_img = after_clean_plate_img
                    return after_check_plate_img, characters_on_plate, coordinates
        return None, None, None



    def find_possible_plates(self, input_img):
        """
        Finding all possible contours that can be plates
        """
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)
        #print(len(possible_plate_contours))
        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)

        if (len(plates) > 0):
            return plates
        else:
            return None

    def find_characters_on_plate(self, plate):

        charactersFound = segment_chars(plate, 400)
        if charactersFound:
            return charactersFound
    
 
    def alternative_segment_chars(self,img):
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        bfilter = cv2.bilateralFilter(gray, 11,17,17) 
        edged= cv2.Canny(bfilter, 30,200) 
       # plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
        location = None
        q=0
        global check
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                 q=1
                 location = approx
                 break
        if q==0:
            check=0
            return 0
        
        print(location)
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0,255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        #plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        #plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        img=cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        global yoyo
        yoyo+=1
        check=1 
       # plt.imsave("C:\\Users\\User\\Documents\\GitHub\\number_plat_github_clone\\Car-License-Plate-Recognition\\image{y}.png".format(y=yoyo),cropped_image)
        return img
        
        
        
    # PLATE FEATURES
    def ratioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 3
        ratioMax = 6

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def preRatioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 2.5
        ratioMax = 7

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect

        if (width > height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle

        if angle > 15:
            return False
        if (height == 0 or width == 0):
            return False

        area = width * height
        if not self.preRatioCheck(area, width, height):
            return False
        else:
            return True





class NeuralNetwork:
    def __init__(self):
        self.model_file = "C:\\Users\\User\\Documents\\GitHub\\number_plat_github_clone\\Car-License-Plate-Recognition\\model\\binary_128_0.50_ver3.pb"
        self.label_file = "C:\\Users\\User\\Documents\\GitHub\\number_plat_github_clone\\Car-License-Plate-Recognition\\model\\binary_128_0.50_labels_ver2.txt"
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def convert_tensor(self, image, imageSizeOuput):
        """
    takes an image and tranform it in tensor
    """
        image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        return np_final

    def label_image(self, tensor):

        input_name = "import/input"
        output_name = "import/final_result"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        results = self.sess.run(output_operation.outputs[0],
                                {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]
        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            plate = plate + self.label_image(self.convert_tensor(img, imageSizeOuput))
        return plate, len(plate)


if __name__ == "__main__":
    findPlate = PlateFinder()

    # Initialize the Neural Network
    model = NeuralNetwork()

    #cap = cv2.VideoCapture('C:\\Users\\User\\Documents\\GitHub\\number_plat_github_clone\\Automatic_NP_Detection_EasyOCR\\render3.mp4')
    cap = cv2.VideoCapture('C:\\Users\\User\\Downloads\\cctv_car_by_me_1.mp4')
    #cap = cv2.VideoCapture('C:\\Users\\User\\Downloads\\cctv_car_by_me3.mp4')
    #cap = cv2.VideoCapture('C:\\Users\\User\\Downloads\\170609_A_Delhi_026.mp4')
    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            cv2.imshow('original video', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            possible_plates = findPlate.find_possible_plates(img)
            if possible_plates==None:
                pre_final_image=findPlate.alternative_segment_chars(img)
                char_on_p=0
                if check==0:
                    continue 
                
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                gray = cv2.cvtColor(pre_final_image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                blur = cv2.medianBlur(thresh, 3)

                # Apply dilation to make the text more clear
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilation = cv2.dilate(blur, kernel, iterations=1)
                clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                # Extract the license plate string using Tesseract
                text3 = pytesseract.image_to_string(clean)
                # Apply pytesseract to extract text from the image
                text2 = pytesseract.image_to_string(dilation, lang='eng')
                text1 = pytesseract.image_to_string(thresh, config='--psm 6')
                count1 = len(''.join(filter(str.isalnum, text1)))
                count2 = len(''.join(filter(str.isalnum, text2)))
                count3 = len(''.join(filter(str.isalnum, text3)))
                if count1 >= 8:
                    char_on_p=1
                    a=(''.join(filter(str.isalnum, text1)))
                    print(a)
                    #a=(''.join(filter(str.isalnum, text1)))
                    #print(a)   
                elif count2 >= 8:
                    char_on_p=1
                    a=(''.join(filter(str.isalnum, text2)))
                    print(a)
                elif count3 >= 8:
                    char_on_p=1
                    a=(''.join(filter(str.isalnum, text3)))
                    print(a)     
                elif min(count1,count2,count3)>=5:
                    char_on_p=1 
                else:
                    char_on_p=0
                if (char_on_p==1):
                    plt.imsave("C:\\Users\\User\\Documents\\GitHub\\number_plat_github_clone\\Car-License-Plate-Recognition\\image_{y}.png".format(y=yoyo),pre_final_image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                        break                   
                
            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)
                    print(recognized_plate)
                    cv2.imshow('plate', p)
                    #cv2.imwrite("C:/Users/use",p)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
