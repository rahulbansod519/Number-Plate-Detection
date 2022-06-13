from ast import Num
from flask import request
from main import *
cap = cv2.VideoCapture('sample4.mp4')
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('https://192.168.43.1:8080/video')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
def generate_frames():
    while True:   
        ret,frame=cap.read()
        if not ret:
            break
        else:
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.8,
                        agnostic_mode=False)

            text = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
            save_results(text)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    rows = Numbers.query.filter_by().all()[0:5]
    return render_template('main.html',rows=rows)

@app.route('/video/')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/main/video/", methods=["GET","POST"])
def main():
    if request.method == "POST":
        date1 = request.form.get('date1')
        date2 = request.form.get('date2')
        if(date1):
            rows = Numbers.query.filter_by(date = date1)
        if(date1 and date2):
            rows = Numbers.query.filter(Numbers.date.between(date1,date2)).all()
    return render_template('main.html',rows=rows)
if __name__=="__main__":
    app.run(debug=True)