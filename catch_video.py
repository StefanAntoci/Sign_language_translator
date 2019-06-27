import cv2
import tensorflow as tf
import numpy as np
import time
import sys
import hand_extract as he 
import neural_network as nn  
total_rectangle = 9

hand_rect_one_x = 0
hand_rect_one_y = 0
hand_rect_two_x = 0
hand_rect_two_y = 0


def get_more_files():
    folder_name = 'E:\\Licenta2019\\Sign_Language_translator\\video_records3'
    files = he.get_all_files(folder_name)
    
    for file in files:
        #print(file)
        splt = file.split('\\')
        new_name = splt[len(splt)-1]+'_'+splt[len(splt)-2]+'.jpg'
        save_folder = 'MoreData'  
        oframe = cv2.imread(file, 1)    
        frame = draw_rect(oframe, False)
        hist = hand_histogram(frame)
        _, img = hist_masking(frame, hist)
        print(folder_name+'\\'+save_folder+'\\'+new_name)
        cv2.imwrite(folder_name+'\\'+save_folder+'\\'+new_name, img)
        

def draw_rect(frame, real_time):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [8 * rows / 20, 8 * rows / 20, 8 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 10 * rows / 20,
         10 * rows / 20, 10 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [10 * cols / 20, 11 * cols / 20, 12 * cols / 20, 10 * cols / 20, 11 * cols / 20, 12 * cols / 20, 10 * cols / 20,
         11 * cols / 20, 12 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    #big_rect_x1 = hand_rect_one_y[0] - 70
    #big_rect_x2 = hand_rect_one_y[2] + 70
    #big_rect_y1 = hand_rect_one_x[0] - 110
    #big_rect_y2 = hand_rect_one_x[6] + 110
    
    if real_time is True:
        for i in range(total_rectangle):
            cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                          (hand_rect_two_y[i], hand_rect_two_x[i]),
                          (0, 255, 0), 1)                  

    return frame

def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)    
    ret, thresh = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))
    return dst, cv2.bitwise_and(thresh, frame)

def load_model(sess, path_to_model, path_to_chkp):
    saver = tf.train.import_meta_graph(path_to_model)
    saver.restore(sess, tf.train.latest_checkpoint(path_to_chkp))
    graph = tf.get_default_graph()
    return graph
    
def predict_conv(graph, sess, image):
    input = graph.get_tensor_by_name("input:0")
    predOut = graph.get_tensor_by_name("predOuts:0")
    feed_dict = { input : image }
    result = sess.run(predOut, feed_dict)
    return result
    
def predict(graph, sess, key_points):
    inPH = graph.get_tensor_by_name("inPH:0")
    predOuts = graph.get_tensor_by_name("predOuts:0")
    feed_dict = {inPH : key_points}
    result = sess.run(predOuts, feed_dict)   
    return result

def get_most_popular_letter(letters):
    letters_app = dict()
    letters_app[letters[0]] = 1
    for i in range(1,len(letters)):
        letter_found = False
        for key, value in letters_app.items():
            if letters[i] == key:
                letters_app[key] = letters_app[key] + 1
                letter_found = True
                break
        if letter_found == False:
            letters_app[letters[i]] = 1
    
    print(letters_app)
    letters = list(letters_app.keys())
    appearances = list(letters_app.values())

    # letter = list(letters_app.keys())[0]
    # appearances = list(letters_app.values())[0]
    # for key, value in letters_app.items():
        # if value > appearances:
            # letter = key
               
    # return letter
    
    if len(letters) == 1:
        return letters[0]
    
    letter1 = letters[0]
    letter2 = letters[1]
    app1 = appearances[0]
    app2 = appearances[1]
    
    if app2 > app1:
        temp = letter2
        letter2 = letter1
        letter1 = temp
        temp = app2
        app2 = app1
        app1 = temp
    
    for x in range(2, len(letters)):
        if letters_app[letters[x]] > app1:
            app2 = app1
            app1 = letters_app[letters[x]]
            letter2 = letter1
            letter1 = letters[x]
        elif letters_app[letters[x]] > app2 and letters_app[letters[x]] < app1:
            app2 = letters_app[letters[x]]
            letter2 = letters[x]
    
    if app1 - app2 >= 2:
        return letter1
    else:
        return ''
    
def capture_video():
    file_name = 0
    real_time = True
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        real_time = False
    with tf.Session() as sess:
        cap = cv2.VideoCapture(file_name)    
        x = 0
        y = 0
        graph = load_model(sess, 'E:\Licenta2019\Sign_Language_translator\\model.ckpt.meta', 'E:\Licenta2019\Sign_Language_translator')
        letters = list()
        frames = list()
        word = ""
        letter = ""
        while(True):           
            ret, frame = cap.read()
            frame = draw_rect(frame, real_time)
            if type(frame) is None:
                break
            hist = hand_histogram(frame)
            prob_img, img = hist_masking(frame, hist)
            contour  = he.get_hand_contour(prob_img)
            if y < 5:
                frames.append(contour)
                y = y + 1
            else:
                filtered_contour = he.delete_noize(frames)
                symbol = he.adjust_hand_contour(filtered_contour)
                frames.clear()
                y = 0
                if len(symbol) != 0:
                    cv2.imshow("symbol", symbol)
                    key_points = he.features_calculation(symbol)
                    areas = he.get_4areas(symbol)
                    if len(key_points) > 1 and len(areas) > 1:
                        for j in range(len(areas)):
                            key_points.append(areas[j])
                        key_points = np.asarray(key_points)
                        key_points = key_points.reshape(1,44)
                        predictions = predict(graph, sess, key_points)
                        predictions_dev = np.std(predictions)
                        letter = nn.prob_to_letter(predictions[0])
                        if x < 5:
                            letters.append(letter)
                            x = x + 1
                        else:
                            letter = get_most_popular_letter(letters)  
                            letters.clear()
                            x = 0                                    
            print(letter)
            cv2.putText(img, letter,(16,450), cv2.FONT_HERSHEY_SIMPLEX,5, (255,0,0), 2, cv2.LINE_AA )                                                     
            cv2.imshow("img", img)
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def create_own_dataset():
    cap = cv2.VideoCapture(0)
    x = 2000
    while(x < 3400):
        ret, frame = cap.read()
        frame = draw_rect(frame)
        hist = hand_histogram(frame)
        prob_img, img = hist_masking(frame, hist)
        cv2.imshow("capturing", img)
        key = cv2.waitKey(1)
        cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myDataset\\K3\\'+'K'+str(x)+'.jpg', img)
        x = x + 1
        print(x)
    cap.release()

#get_more_files()    
capture_video()
#create_own_dataset()