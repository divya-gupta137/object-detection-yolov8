import random
import time

class BLEScanner:
    def __init__(self, target_device_name="UserPhone"):
        """
        Initialize BLE scanner.
        target_device_name: Name of the BLE device to follow
        """
        self.target_name = target_device_name

    def get_mock_signal(self):

        rssi = random.randint(-80, -40)  # Simulate distance
        if rssi > -50:
            direction = "FORWARD"
        elif rssi > -65:
            direction = "LEFT"
        else:
            direction = "RIGHT"
        return {"rssi": rssi, "direction": direction}

    def get_real_signal(self):
        """
        Placeholder for real BLE scanning logic on Raspberry Pi.
        You can later use 'bleak' or 'bluepy' to scan devices.
        Should return the same dict format: {'rssi': value, 'direction': ...}
        """
        # Example template
        # from bleak import BleakScanner
        # devices = await BleakScanner.discover()
        # for d in devices:
        #     if d.name == self.target_name:
        #         rssi = d.rssi
        #         # Map RSSI to direction
        #         ...
        return {"rssi": -60, "direction": "FORWARD"}  # Placeholder

# Example usage
if __name__ == "__main__":
    scanner = BLEScanner()
    
    while True:
        # For now, using mock signal
        signal = scanner.get_mock_signal()
        print("BLE Signal:", signal)
        time.sleep(1)
