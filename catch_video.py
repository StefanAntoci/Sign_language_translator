import cv2
import tensorflow as tf
import numpy as np
import time
import hand_extract as he 
import neural_network as nn  
total_rectangle = 9

hand_rect_one_x = 0
hand_rect_one_y = 0
hand_rect_two_x = 0
hand_rect_two_y = 0

def draw_rect(frame):
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

    big_rect_x1 = hand_rect_one_y[0] - 70
    big_rect_x2 = hand_rect_one_y[2] + 70
    big_rect_y1 = hand_rect_one_x[0] - 110
    big_rect_y2 = hand_rect_one_x[6] + 110
    
    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)
    #cv2.rectangle(frame, (big_rect_x1, big_rect_y1), (big_rect_x2, big_rect_y2), (0,255,0), 1)                  

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
            
    letter = list(letters_app.keys())[0]
    appearances = list(letters_app.values())[0]
    for key, value in letters_app.items():
        if value > appearances:
            letter = key
    print(letters_app)            
    return letter
    
def capture_video():
    with tf.Session() as sess:
        cap = cv2.VideoCapture(0)    
        x = 0
        y = 0
        graph = load_model(sess, 'E:\Licenta2019\Sign_Language_translator\\model.ckpt.meta', 'E:\Licenta2019\Sign_Language_translator')
        letters = list()
        frames = list()
        while(True):
            ret, frame = cap.read()
            frame = draw_rect(frame)
            hist = hand_histogram(frame)
            prob_img, img = hist_masking(frame, hist)
            cv2.imshow("img", img)
            contour  = he.get_hand_contour(prob_img)
            if y < 5:
                frames.append(contour)
                y = y + 1
            else:
                filtered_contour = he.delete_noize(frames)
                symbol = he.adjust_hand_contour(filtered_contour)
                cv2.imshow("symbol", symbol)
                frames.clear()
                y = 0
                
                key_points = he.features_calculation(symbol)
                if len(key_points) > 1:
                    key_points = np.asarray(key_points)
                    key_points = key_points.reshape(1,40)           
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
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def capture_video_conv():
    with tf.Session() as sess:
        cap = cv2.VideoCapture(0)
        graph = load_model(sess, 'E:\\Licenta2019\\Sign_Language_translator\\conv_model\\ConvModel.ckpt.meta', 'E:\\Licenta2019\\Sign_Language_translator\\conv_model')
        x = 0
        letters = list()
        frames = list()
        while(True):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (200, 200) )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis = 2)
            frames.append(frame)
            prediction = predict_conv(graph, sess, frames)
            print(prediction)
            frames.clear()
            letter = nn.prob_to_letter_conv(prediction[0])
            # if x < 5:
                # letters.append(letter)
                # x = x + 1
            # else:
                # letter = get_most_popular_letter(letters)
                # letters.clear()
                # x = 0
            print(letter)
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()       

def create_own_dataset():
    cap = cv2.VideoCapture(0)
    x = 0
    while(x < 1000):
        ret, frame = cap.read()
        frame = draw_rect(frame)
        hist = hand_histogram(frame)
        prob_img, img = hist_masking(frame, hist)
        cv2.imshow("capturing", img)
        key = cv2.waitKey(1)
        cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myDataset\\C\\'+'C'+str(x)+'.jpg', img)
        x = x + 1
        print(x)
    cap.release()
    
capture_video()
#create_own_dataset()