from subprocess import call
from datetime import datetime

if __name__ == "__main__":
    configs = [
        "izmit_5_150",
        "izmit_5_175",
        "izmit_5_200",
        "izmit_5_225",
        "izmit_5_250",
        "izmit_6_150",
        "izmit_6_175",
        "izmit_6_200",
        "izmit_6_225",
        "izmit_6_250",
        "izmit_7_150",
        "izmit_7_175",
        "izmit_7_200",
        "izmit_7_225",
        "izmit_7_250",
        "izmit_8_150",
        "izmit_8_175",
        "izmit_8_200",
        "izmit_8_225",
        "izmit_8_250",
    ]
    
    for config in configs:
        print("-" * 80)
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}: RUNNING CONFIG: {config}")
        call(["oplrun", "-p", ".", config])
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}: END CONFIG: {config}")
        print("-" * 80)