import requests

def send_gcode_script(gcode_script, moonraker_url='http://192.168.1.119:7125'):
    print(gcode_script)
    """
    Sends a G-code script to the Moonraker API server.

    :param gcode_script: The G-code script to send.
    :param moonraker_url: The base URL of the Moonraker API server.
    """
    url = f"{moonraker_url}/printer/gcode/script"
    headers = {'Content-Type': 'application/json'}
    data = {'script': gcode_script}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending the G-code script: {e}")
        if e.args[0].startswith('400 Client Error: Must home axis first'):
            send_gcode_script('G28')
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            return response.json()
        else:
            raise e

def standby():
    send_gcode_script(f"G0 Z30 F10000")
    send_gcode_script("G0 X0 Y350 Z100 F20000")