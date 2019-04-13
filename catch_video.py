import cv2
import tensorflow as tf
import numpy as np
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
    ret, thresh = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))
    #return cv2.bitwise_and(thresh, frame)
    return dst, cv2.bitwise_and(thresh, frame)

def load_model(sess):
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    return graph
    
def predict(graph, sess, key_points):
    inPH = graph.get_tensor_by_name("inPH:0")
    predOuts = graph.get_tensor_by_name("predOuts:0")
    feed_dict = {inPH : key_points}
    result = sess.run(predOuts, feed_dict)   
    return result

    
def capture_video():

    with tf.Session() as sess:
        cap = cv2.VideoCapture(0)    
        x = 0
        graph = load_model(sess)
        while(True):
            x = x + 1
            ret, frame = cap.read()
            
            frame = draw_rect(frame)
            hist = hand_histogram(frame)
            prob_img, img = hist_masking(frame, hist)
            
            #roi = he.hsv_mask(frame)
            
            #symbol, cnt= he.hand_extraction(roi)
            symbol, cnt = he.hand_extraction(prob_img)
            cv2.imshow("symbol", symbol)
            cv2.imshow("img", img)
            #print(cnt)
            if len(cnt) >= 20:
                key_points = he.features_calculation(symbol)
                key_points = np.asarray(key_points)
                key_points = key_points.reshape(1,20)
                #print(key_points)            
                predictions = predict(graph, sess, key_points)
                predictions_dev = np.std(predictions)
                #print(predictions)
                #if predictions_dev > 15:
                letter = nn.prob_to_letter(predictions[0])
                print(letter)            
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
capture_video()
