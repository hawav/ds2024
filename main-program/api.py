from flask import Flask, jsonify, request
from flask_cors import CORS
from printer import send_gcode_script
from main import show_video_stream, sm, SQUARES, Status
from util import SquareState
from prepare_gaming import check_game_ready
from monitor_gaming import verify_movement
import threading

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def weijun():
    return "Hello，这里是三子棋服务端 :)"

@app.get('/status')
def get_status():
    return str(sm.status)

@app.get('/board')
def get_board():
    return jsonify(sm.board_status.squares.tolist())

@app.route('/home', methods=['GET'])
def home():
    # send_gcode_script("G90")
    send_gcode_script("G0 X0 Y350 Z100 F20000")
    return jsonify({"message": "Going Home"})

@app.route('/put', methods=['GET'])
def put():
    idx = request.args.get('idx', type=int)
    if idx >= 9:
        print(idx, '超出位置')
        return '超出位置'
    # send_gcode_script("G90")
    
    side = None
    if request.args.get('black', type=str) == 'true':
        side = SquareState.Black
    else:
        side = SquareState.White
    sm.handle_event('put', idx=idx, side=side)
    
    return 'OK'

@app.route('/game/start', methods=['GET'])
def game_start():
    side = request.args.get('side', type=str)
    sm.side = SquareState.Black if side == 'black' else SquareState.White
    sm.next_square = request.args.get('pos', type=int)
    sm.status = Status.PREPARE_GAMING

    err = check_game_ready(sm, SQUARES)
    if err is None:
        return 'OK'
    else:
        return err

@app.route('/game/next', methods=['GET'])
def game_next():
    err = verify_movement(sm)

    if err is not None:
        return err

    sm.handle_event('their_turn_finished')
    return 'OK'

@app.route('/game/ready')
def game_ready():
    err = check_game_ready(sm, SQUARES)
    if err is None:
        return 'OK'
    else:
        return err
    
@app.route('/game/reset')
def game_reset():
    sm.handle_event('reset')
    return 'OK'

def vision():
    stream_url = "http://192.168.1.119/webcam/?action=stream"
    while True:
        try:
            show_video_stream(stream_url, 0, 0)
        finally:
            pass

if __name__ == '__main__':
    v = threading.Thread(target=vision)
    v.start()
    app.run(host='0.0.0.0', port=5000)
