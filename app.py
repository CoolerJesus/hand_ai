from flask import Flask, render_template, Response, request, jsonify, session
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'sign_language_secret_key'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

model = None
labels = []

def load_model():
    global model, labels
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        if os.path.exists('labels.pkl'):
            with open('labels.pkl', 'rb') as f:
                labels = pickle.load(f)
        return True
    except:
        return False

def normalize_landmarks(landmarks, wrist):
    normalized = []
    for lm in landmarks:
        normalized.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z])
    return normalized

@app.route('/')
def index():
    has_model = load_model()
    stats = get_stats()
    return render_template('index.html', has_model=has_model, stats=stats, labels=labels)

@app.route('/collect')
def collect_page():
    return render_template('collect.html')

@app.route('/video')
def video():
    if not model:
        return "No model", 400
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prediction_buffer = []
    current_sentence = session.get('sentence', [])
    
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            all_landmarks = np.zeros(126)
            
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if i > 1:
                        break
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    wrist = hand_landmarks.landmark[0]
                    coords = normalize_landmarks(hand_landmarks.landmark, wrist)
                    all_landmarks[i*63:(i+1)*63] = coords
                
                try:
                    prediction = model.predict([all_landmarks])[0]
                    prob = np.max(model.predict_proba([all_landmarks]))
                    
                    if prob > 0.75:
                        prediction_buffer.append(prediction)
                        if len(prediction_buffer) > 10:
                            prediction_buffer.pop(0)
                        
                        from collections import Counter
                        most_common = Counter(prediction_buffer).most_common(1)[0]
                        if most_common[1] >= 8:
                            if not current_sentence or prediction != current_sentence[-1]:
                                current_sentence.append(prediction)
                                session['sentence'] = current_sentence[-20:]
                
                except:
                    pass
            
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h-60), (w, h), (20, 20, 20), -1)
            
            sentence_str = " ".join(current_sentence[-8:]) if current_sentence else "..."
            cv2.putText(frame, f"Sentence: {sentence_str}", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if model:
                try:
                    pred = model.predict([all_landmarks])[0]
                    prob = np.max(model.predict_proba([all_landmarks]))
                    cv2.putText(frame, f"{pred} ({prob:.0%})", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except:
                    pass
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record', methods=['POST'])
def record():
    sign_name = request.form.get('sign_name', '').strip()
    if not sign_name:
        return jsonify({'success': False, 'error': 'Name required'})
    
    cap = cv2.VideoCapture(0)
    count = 0
    target = 50
    
    while count < target:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                wrist = hand_lms.landmark[0]
                coords = normalize_landmarks(hand_lms.landmark, wrist)
                row = coords + [sign_name]
                
                df = pd.DataFrame([row])
                df.to_csv('podaci.csv', mode='a', index=False, header=False)
                count += 1
        
        cv2.putText(frame, f"Recording: {sign_name}", (50, 50), 1, 2, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {count}/{target}", (50, 100), 1, 2, (255, 255, 0), 2)
        cv2.imshow("Recording", frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return jsonify({'success': True, 'message': f'Recorded {count} samples for "{sign_name}"'})

@app.route('/train', methods=['POST'])
def train():
    if not os.path.exists('podaci.csv'):
        return jsonify({'success': False, 'error': 'No data found'})
    
    try:
        data = pd.read_csv('podaci.csv', header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        from sklearn.ensemble import RandomForestClassifier
        global model, labels
        model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        model.fit(X, y)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        labels = list(np.unique(y))
        with open('labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
        
        score = model.score(X, y)
        return jsonify({'success': True, 'accuracy': f'{score*100:.1f}%', 'labels': labels})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear', methods=['POST'])
def clear():
    session['sentence'] = []
    return jsonify({'success': True})

@app.route('/stats')
def get_stats():
    try:
        if os.path.exists('podaci.csv'):
            df = pd.read_csv('podaci.csv', header=None)
            return {
                'samples': len(df),
                'signs': df.iloc[:, -1].nunique(),
                'details': df.iloc[:, -1].value_counts().to_dict()
            }
    except:
        pass
    return {'samples': 0, 'signs': 0, 'details': {}}

@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  AI Sign Language Translator - Web Version")
    print("="*50)
    print("  Open: http://localhost:5000")
    print("  Works on: Desktop & Mobile browsers")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
