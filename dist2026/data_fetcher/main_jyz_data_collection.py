import time

import requests

from bitable import Bitable
import os

from config_util import Config

config = Config(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.ini",
    )
)

BASE_URL = config.get_config("DEFAULT", "BASE_URL")
username = config.get_config("DEFAULT", "username")
password = config.get_config("DEFAULT", "password")

LOGIN_DATA = {
    "username": username,
    "password": password
}

collect_all_devices = config.get_config_bool("QUERY", "collect_all_devices")
table_id = config.get_config("DEFAULT", "table_id")
app_id = config.get_config("DEFAULT", "app_id")
app_secret = config.get_config("DEFAULT", "app_secret")
wiki_id = config.get_config("DEFAULT", "wiki_id")


def time_convert(t):
    """
    Converts time between Unix timestamp in milliseconds and a formatted string.

    Parameters:
    - t (int or str): If t is an integer, it is assumed to be a Unix timestamp in milliseconds
                      and will be converted to a formatted string "%Y-%m-%d_%H:%M:%S".
                      If t is a string, it is assumed to be in the format "%Y-%m-%d_%H:%M:%S"
                      and will be converted to a Unix timestamp in milliseconds.

    Returns:
    - int or str: The converted time, either as a Unix timestamp in milliseconds or a formatted string.
    """
    if isinstance(t, int):
        tmp = t / 1000
        time_local = time.localtime(tmp)
        ans = time.strftime("%Y-%m-%d_%H:%M:%S", time_local)
    else:
        tmp = time.strptime(t, "%Y-%m-%d_%H:%M:%S")
        ans = int(time.mktime(tmp)) * 1000

    return ans


def login():
    """
    Logs in to the system and returns a session object if successful.

    Parameters:
    - None

    Returns:
    - requests.Session or None: A session object if login is successful, otherwise None.
    """
    session = requests.Session()
    response = session.post(f"{BASE_URL}/login", data=LOGIN_DATA)
    if response.url.endswith('/index'):
        print("Login successful!")
        return session
    else:
        print("Login failed.")
        return None


def get_imei_numbers(session):
    """
    Retrieves a list of IMEI numbers using a session object.

    Parameters:
    - session (requests.Session): A session object used to make the request.

    Returns:
    - list: A list of IMEI numbers if the request is successful, otherwise an empty list.
    """

    response = session.get(f"{BASE_URL}/get_imei_numbers")
    if response.status_code == 200:
        return response.json()['imeiNumbers']
    else:
        print("Failed to get IMEI numbers")
        return []


def get_device_config(session, imei):
    """
    Retrieves the configuration for a specific device identified by its IMEI number.

    Parameters:
    - session (requests.Session): An authenticated session object used to make the request.
    - imei (str): The International Mobile Equipment Identity (IMEI) number of the device.

    Returns:
    - dict or None: A dictionary containing the device configuration if the request is successful,
                    otherwise None if the request fails.
    """
    response = session.get(f"{BASE_URL}/device/{imei}/config")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get config for IMEI {imei}")
        return None


def get_runtimes(session, imei):
    """
    Retrieves the runtimes for a specific device identified by its IMEI number.

    Parameters:
    - session (requests.Session): An authenticated session object used to make the request.
    - imei (str): The International Mobile Equipment Identity (IMEI) number of the device.

    Returns:
    - dict or None: A dictionary containing the device runtimes if the request is successful,
                   otherwise None if the request fails.
    """
    payload = {
        "selected_imei": imei
    }
    response = session.post(f"{BASE_URL}/get_runtimes", data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get runtimes for IMEI {imei}")
        return None


def get_timed_status(session, imei, runtime):
    """
    Retrieves the status of a device at a specific runtime.

    Parameters:
    - session (requests.Session): An authenticated session object used to make the request.
    - imei (str): The International Mobile Equipment Identity (IMEI) number of the device.
    - runtime (str): The runtime for which to retrieve the status.

    Returns:
    - dict or None: A dictionary containing the device status at the specified runtime if the request is successful,
                    otherwise None if the request fails.
    """
    response = session.get(f"{BASE_URL}/get_timed_status/{imei}/{runtime}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get timed status for IMEI {imei}")
        return None


def upload_to_target(session, target, index, devices, start_time, end_time):
    """
    Uploads device data to a target system within a specified time range.

    Parameters:
    - session (requests.Session): An authenticated session object used to make requests for device data.
    - target: The target system where the data will be uploaded. This should have a method 'process_post_data'.
    - index (int): The index or identifier for the target system.
    - devices (list): A list of IMEI numbers for the devices to upload data from.
    - start_time (int): The Unix timestamp in milliseconds representing the start of the time range.
    - end_time (int): The Unix timestamp in milliseconds representing the end of the time range.

    Raises:
    - Exception: If the start time is invalid (less than 0).

    This function iterates over a list of devices, retrieves their runtimes, and uploads the data to the target system
    if the runtime falls within the specified time range. It also formats the data and handles batch creation on the target system.
    """
    if start_time <= 0:
        raise Exception("\n\n\n\nInvalid start time! Start time smaller than 0.")

    for imei in devices:
        runtimes = get_runtimes(session, imei)
        print(f"Uploading for device #{imei}...")
        record_list = {"records": []}

        for runtime in runtimes:
            unix = time_convert(runtime)
            if start_time <= unix <= end_time:
                print(runtime)
                status = get_timed_status(session, imei, runtime)
                analog = float(status["Analog"])
                battery = float(status["Battrey"])
                humidity = float(status["Humidity"])
                version = status["LocalVersion"]
                temperature = float(status["Temperature"])
                analog = "{:.4f}".format(analog)
                battery = "{:.3f}".format(battery)
                humidity = "{:.3f}".format(humidity)
                temperature = "{:.3f}".format(temperature)
                content = [str(imei), unix, str(battery), str(analog), str(temperature), str(humidity), str(version)]
                print(content)
                record = target.process_post_data(index, content)
                record_list["records"].append(record)

        target.batch_create(record_list)
        print("UPLOAD COMPLETED!\n")
        input("QUESTION ")

def main():
    """
    Main function to handle the login, data retrieval, and upload process.

    This function performs the following steps:
    1. Logs into the system and starts a session.
    2. Retrieves a list of IMEI numbers associated with the session.
    3. Sets up the index and data types for the Bitable to be created.
    4. Initializes the Bitable target with the provided credentials and wiki ID.
    5. Converts the start and end times from configuration to the appropriate format.
    6. Retrieves the list of devices from the configuration.
    7. Creates a new table in Bitable with the specified name, index, and data types.
    8. Depending on the configuration, uploads data for all devices or a subset of devices.
    9. Handles any exceptions that occur during the upload process.
    10. Closes the session upon completion or error.

    Parameters:
    None

    Returns:
    None
    """
    session = login()

    if not session:
        session.close()

    imei_numbers = get_imei_numbers(session)

    index = ["IMEI", "时间", "电池电压", "测量值", "温度", "湿度", "版本号"]
    data_types = ["文本", "日期", "文本", "文本", "文本", "文本", "文本"]

    target = Bitable(table_id, app_id, app_secret, wiki_id)

    start_time = config.get_config("QUERY", "start_time")
    start_time = time_convert(start_time)

    end_time = config.get_config("QUERY", "end_time")
    end_time = time_convert(end_time)

    devices = config.get_config_list("QUERY", "devices")
    name = config.get_config("QUERY", "table_name")
    # print(int(collect_all_devices))
    new_table = target.create_new_table(name, index, data_types)
    try:
        if collect_all_devices:
            upload_to_target(session, new_table, index, imei_numbers, start_time, end_time)
        else:
            print("Processing...")
            upload_to_target(session, new_table, index, devices, start_time, end_time)
        input("Uploading completed. Press any button to exit...")
    except Exception as e:
        print(e)
        input("Uploading incomplete. Please check. Press any button to exit...")

    session.close()


main()
