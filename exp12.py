"""
Ex. No: 12 - AUTOMATED STUDENT VERIFICATION AND ATTENDANCE LOGGING SYSTEM USING DEEP LEARNING COMPUTER VISION TECHNIQUES
"""
import datetime
import os
import tkinter as tk
from tkinter import ttk
import numpy as np

STUDENT_DATABASE = {
    "732524AD034": {
        "name": "MAHESHKUMAR",
        "class": "B.TECH AI&DS",
        "dummy_feature_vector": np.ones((128,), dtype=np.float32),
    }
}

class DeepLearningAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SVHEC CC3 LAB ATTENDANCE - Automated Student Attendance Verification System")
        self.root.geometry("1120x680")
        self.root.configure(bg="#F4F6F9")

        self.total_pcs = 66
        self.occupied_pcs = 0
        self.checkins_today = 1
        self.scan_mode = tk.StringVar(value="ENTRY")
        self.storage_path = r"E:\Attendance-System\Attendance-System\data"

        self._build_header()
        self._build_main_layout()
        self._populate_initial_log()
        self._update_clock()

    def _build_header(self):
        header = tk.Frame(self.root, bg="#FFFFFF", height=60, bd=1, relief="solid")
        header.pack(side="top", fill="x", padx=5, pady=5)

        title_frame = tk.Frame(header, bg="#FFFFFF")
        title_frame.pack(side="left", padx=10, pady=5)
        tk.Label(title_frame, text="SVHEC CC3 LAB ATTENDANCE", font=("Segoe UI", 14, "bold"), fg="#1E7E34", bg="#FFFFFF").pack(anchor="w")
        tk.Label(title_frame, text="Automated Student Attendance Verification System (ENTRY / OUT Modes)", font=("Segoe UI", 8), fg="#555555", bg="#FFFFFF").pack(anchor="w")

        tk.Label(header, text=f" Storage: {self.storage_path}", font=("Consolas", 8), bg="#E9ECEF", fg="#333333", relief="groove", padx=6, pady=2).pack(side="left", padx=20)

        self.clock_frame = tk.Frame(header, bg="#FFFFFF")
        self.clock_frame.pack(side="right", padx=15)
        self.time_label = tk.Label(self.clock_frame, font=("Segoe UI", 13, "bold"), fg="#0A58CA", bg="#FFFFFF")
        self.time_label.pack(anchor="e")
        self.date_label = tk.Label(self.clock_frame, font=("Segoe UI", 8), fg="#666666", bg="#FFFFFF")
        self.date_label.pack(anchor="e")

    def _build_main_layout(self):
        content = tk.Frame(self.root, bg="#F4F6F9")
        content.pack(fill="both", expand=True, padx=5, pady=2)

        # LEFT COLUMN - SCANNER INTERFACE
        left_col = tk.Frame(content, bg="#FFFFFF", bd=1, relief="solid", width=480)
        left_col.pack(side="left", fill="both", padx=5, pady=5)

        scan_header = tk.Frame(left_col, bg="#FFFFFF")
        scan_header.pack(fill="x", padx=10, pady=8)
        tk.Label(scan_header, text="ID CARD SCANNER", font=("Segoe UI", 10, "bold"), bg="#FFFFFF").pack(side="left")
        tk.Label(scan_header, text="Camera: ON", bg="#0D6EFD", fg="white", font=("Segoe UI", 8, "bold"), padx=5).pack(side="right")
        tk.Label(scan_header, text="Live: 30 FPS", fg="#0DCAF0", bg="#FFFFFF", font=("Segoe UI", 8)).pack(side="right", padx=8)

        # Mode Selector
        mode_bar = tk.Frame(left_col, bg="#FFFFFF")
        mode_bar.pack(fill="x", padx=10, pady=4)
        tk.Label(mode_bar, text="SCAN MODE: ", font=("Segoe UI", 8, "bold"), bg="#FFFFFF").pack(side="left")
        tk.Radiobutton(mode_bar, text="● 1. ENTRY", variable=self.scan_mode, value="ENTRY", indicatoron=0, bg="#198754", fg="white", selectcolor="#157347", font=("Segoe UI", 8, "bold"), padx=8, command=self._on_mode_change).pack(side="left", padx=4)
        tk.Radiobutton(mode_bar, text="○ 2. OUT", variable=self.scan_mode, value="OUT", indicatoron=0, bg="#E9ECEF", fg="#333", selectcolor="#CED4DA", font=("Segoe UI", 8, "bold"), padx=8, command=self._on_mode_change).pack(side="left", padx=4)

        # Camera Viewport Box
        self.cam_label = tk.Label(left_col, bg="#000000", height=15)
        self.cam_label.pack(fill="x", padx=10, pady=10)

        # Scan Prompt Status Box
        status_card = tk.Frame(left_col, bg="#FFFFFF", bd=1, relief="groove")
        status_card.pack(fill="both", expand=True, padx=10, pady=10)
        self.status_prompt = tk.Label(status_card, text=" \nREADY TO SCAN [ENTRY MODE]", font=("Segoe UI", 11, "bold"), fg="#212529", bg="#FFFFFF")
        self.status_prompt.pack(expand=True)

        tk.Button(status_card, text="⚡ Simulate Deep Learning Verification", bg="#0D6EFD", fg="white", font=("Segoe UI", 8, "bold"), command=self._simulate_deep_learning_verification).pack(pady=4)

        # RIGHT COLUMN - STATS & LOG TABLE
        right_col = tk.Frame(content, bg="#F4F6F9")
        right_col.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        metrics_frame = tk.Frame(right_col, bg="#F4F6F9")
        metrics_frame.pack(fill="x", pady=2)
        self.lbl_checkins = self._build_stat_card(metrics_frame, "CHECK-INS TODAY", str(self.checkins_today), "#0D6EFD")
        self.lbl_occupied = self._build_stat_card(metrics_frame, "OCCUPIED PCS", f"{self.occupied_pcs} / {self.total_pcs}", "#198754")
        self.lbl_available = self._build_stat_card(metrics_frame, "AVAILABLE PCS", str(self.total_pcs - self.occupied_pcs), "#FD7E14")

        table_card = tk.Frame(right_col, bg="#FFFFFF", bd=1, relief="solid")
        table_card.pack(fill="both", expand=True, pady=6)
        tk.Label(table_card, text="TODAY'S ATTENDANCE LOG", font=("Segoe UI", 9, "bold"), bg="#FFFFFF").pack(anchor="w", padx=10, pady=6)

        columns = ("Time", "Reg No", "Student Name", "Class", "PC / Status")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", height=12)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")
        self.tree.column("Time", width=140)
        self.tree.column("Reg No", width=110)
        self.tree.column("Student Name", width=120)
        self.tree.column("Class", width=100)
        self.tree.column("PC / Status", width=140)
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Action Toolbar
        actions_frame = tk.Frame(right_col, bg="#F4F6F9")
        actions_frame.pack(fill="x", pady=4)
        r1 = tk.Frame(actions_frame, bg="#F4F6F9")
        r1.pack(fill="x", pady=2)
        tk.Button(r1, text="Generate Attendance Report", bg="#00AA6C", fg="white", font=("Segoe UI", 9, "bold")).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(r1, text="Open Data Folder", bg="#1A3B5C", fg="white", font=("Segoe UI", 8)).pack(side="left", padx=2)
        tk.Button(r1, text="Open Excel Log", bg="#1A3B5C", fg="white", font=("Segoe UI", 8)).pack(side="left", padx=2)

    def _build_stat_card(self, parent, title, value, color):
        card = tk.Frame(parent, bg="#FFFFFF", bd=1, relief="solid", padx=10, pady=5)
        card.pack(side="left", fill="x", expand=True, padx=3)
        tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg="#555", bg="#FFFFFF").pack()
        lbl_val = tk.Label(card, text=value, font=("Segoe UI", 16, "bold"), fg=color, bg="#FFFFFF")
        lbl_val.pack()
        return lbl_val

    def _populate_initial_log(self):
        self.tree.insert("", "end", values=("06:20:30 - 06:20:47", "732524AD034", "MAHESHKUMAR", "B.TECH AI&DS", "PC-08 (COMPLETED)"))

    def _simulate_deep_learning_verification(self):
        detected_reg_no = "732524AD034"
        record = STUDENT_DATABASE.get(detected_reg_no)
        if record:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            mode = self.scan_mode.get()
            if mode == "ENTRY":
                self.occupied_pcs = min(self.total_pcs, self.occupied_pcs + 1)
                self.checkins_today += 1
                status = f"PC-0{self.occupied_pcs} (ACTIVE)"
            else:
                self.occupied_pcs = max(0, self.occupied_pcs - 1)
                status = "PC-08 (COMPLETED)"
            
            self.tree.insert("", 0, values=(current_time, detected_reg_no, record["name"], record["class"], status))
            self.lbl_checkins.config(text=str(self.checkins_today))
            self.lbl_occupied.config(text=f"{self.occupied_pcs} / {self.total_pcs}")
            self.lbl_available.config(text=str(self.total_pcs - self.occupied_pcs))

    def _on_mode_change(self):
        self.status_prompt.config(text=f" \nREADY TO SCAN [{self.scan_mode.get()} MODE]")

    def _update_clock(self):
        now = datetime.datetime.now()
        self.time_label.config(text=now.strftime("%H:%M:%S"))
        self.date_label.config(text=now.strftime("%A, %d %B %Y"))
        self.root.after(1000, self._update_clock)

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepLearningAttendanceApp(root)
    root.mainloop()

