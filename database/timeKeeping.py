import os
from datetime import datetime
import pandas as pd
from firebase_admin import db



# Lấy dữ liệu từ Firebase
def get_data(bucket_name):
    ref = db.reference(bucket_name)
    data = ref.get()
    return data

# Lưu dữ liệu vào Firebase
def save_data(bucket_name, data):
    ref = db.reference(bucket_name)
    ref.update(data)
    print("Data updated in Firebase")

    
def create_daily_timekeeping(bucket_name, date=None):

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Check if timekeeping data for this date already exists
    existing_data = get_data(f"{bucket_name}/Timekeeping/{date}")
    if existing_data is not None:
        print(f"Timekeeping table for {date} already exists!")
        return False

    # Get all employees
    employees = get_data(f"{bucket_name}/Employees")
    if not employees:
        print("No employees found in Employees bucket!")
        return False

    # Create default values for all employees
    daily_data = {}
    for employee_id in employees.keys():
        daily_data[employee_id] = {
            "check_in": "",
            "check_out": "",
            "working_hours": 0.0,
            "attendance_time": 0.0,
            "attendance_in": "",
            "attendance_out": "",
            "comes_late": 0.0,
            "leaves_early": 0.0,
            "overtime": 0.0
        }

    # Save to bucket Timekeeping/YYYY-MM-DD
    save_data(f"{bucket_name}/Timekeeping/{date}", daily_data)
    print(f"Created timekeeping table for {date}")
    return True

# Function to round up to the nearest 30 minutes for attendance_in
def round_up_to_nearest_30_minutes(timestamp):
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    minute = dt.minute
    # If minutes > 0 and <= 30, round up to 30 minutes
    if 0 < minute <= 30:
        minute = 30
    # If minutes > 30, round up to the next hour
    elif minute > 30:
        hour += 1
        minute = 0
    else:
        minute = 0
    return dt.replace(hour=hour, minute=minute, second=0).strftime("%Y-%m-%d %H:%M:%S")

# Function to round down to the nearest hour for attendance_out
def round_down_to_nearest_hour(timestamp):
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return dt.replace(minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")


def process_check_in_out(bucket_name, employee_id, timestamp=None):
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    date = timestamp.split(" ")[0]

    if employee_id == "UNKNOWN":
        unknown_ref = db.reference(f"{bucket_name}/Timekeeping/{date}/UNKNOWN")
        unknown_ref.push(timestamp)

        print(f"🕒 Pushed timestamp to UNKNOWN: {timestamp}")
        return True

    
    ref = db.reference(f"{bucket_name}/Timekeeping/{date}/{employee_id}")
    existing_data = ref.get()

    if not existing_data:
        print(f"❌ No timekeeping record found for employee {employee_id} on {date}. Please create daily timekeeping first.")
        return False

    # Trường hợp chưa check-in
    if existing_data.get("check_in", "") == "":
        ref.update({"check_in": timestamp})
        print(f"[CHECK-IN] ✅ Employee {employee_id} checked in at {timestamp}")
        return "check_in"

    # Trường hợp đã check-in (có hoặc chưa check-out) → luôn update lại check-out
    ref.update({"check_out": timestamp})

    # Tính toán lại tất cả thông số
    check_in_time = datetime.strptime(existing_data["check_in"], "%Y-%m-%d %H:%M:%S")
    check_out_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    working_hours = (check_out_time - check_in_time).total_seconds() / 3600

    attendance_in = round_up_to_nearest_30_minutes(existing_data["check_in"])
    attendance_out = round_down_to_nearest_hour(timestamp)

    attendance_in_time = datetime.strptime(attendance_in, "%Y-%m-%d %H:%M:%S")
    attendance_out_time = datetime.strptime(attendance_out, "%Y-%m-%d %H:%M:%S")
    attendance_time = max(0, (attendance_out_time - attendance_in_time).total_seconds() / 3600)

    start_time = datetime.strptime(f"{date} 08:00:00", "%Y-%m-%d %H:%M:%S")
    comes_late = max(0, (check_in_time - start_time).total_seconds() / 3600)

    end_time = datetime.strptime(f"{date} 17:00:00", "%Y-%m-%d %H:%M:%S")
    leaves_early = max(0, (end_time - check_out_time).total_seconds() / 3600)
    overtime = max(0, (check_out_time - end_time).total_seconds() / 3600)

    ref.update({
        "working_hours": round(working_hours, 2),
        "attendance_in": attendance_in,
        "attendance_out": attendance_out,
        "attendance_time": round(attendance_time, 2),
        "comes_late": round(comes_late, 2),
        "leaves_early": round(leaves_early, 2),
        "overtime": round(overtime, 2)
    })

    print(f"[CHECK-OUT] 🔁 Employee {employee_id} re-checked out at {timestamp}. Working hours: {working_hours:.2f}h")
    return "check_out"





def export_to_excel(bucket_name, date):
    # Path to node Timekeeping/YYYY-MM-DD
    timekeeping_data = get_data(f"{bucket_name}/Timekeeping/{date}")
    
    if not timekeeping_data:
        print(f"No timekeeping data found for {date}")
        return None

    # Prepare data for export
    export_data = []
    for employee_id, record in timekeeping_data.items():
        if employee_id == "UNKNOWN":
            continue
        export_data.append({
            "Employee ID": employee_id,
            "Check In": record.get("check_in", ""),
            "Check Out": record.get("check_out", ""),
            "Working Hours": record.get("working_hours", 0.0),
            "Attendance In": record.get("attendance_in", ""),
            "Attendance Out": record.get("attendance_out", ""),
            "Time Attendance": record.get("attendance_time", 0.0),
            "comes_late": record.get("comes_late", 0.0),
            "Leaves Early": record.get("leaves_early", 0.0),
            "Overtime": record.get("overtime", 0.0)
        })

    # Create DataFrame from data
    df = pd.DataFrame(export_data)
    return df

if __name__ == "__main__":

    # process_check_in_out('Hust', "000000")
    create_daily_timekeeping('Hust')
    # export_to_excel('Hust', '2025-04-14')

