1. Device Name
Motion Analysis System for Rehabilitation (MAS-Rehab)
Software-based motion assessment tool using monocular camera input.

2. Intended Purpose (Annex I – GSPRs)
This software is intended to analyze human musculoskeletal movements using a standard camera for the purposes of:
Monitoring rehabilitation exercises
Counting repetitions
Estimating range of motion (ROM)
Providing real-time corrective feedback on movement quality
The system does not provide diagnosis, prognosis, or automated clinical decisions.
Its intended users are:
Physical therapists
Rehabilitation clinicians
Patients under professional supervision
General users performing guided physical therapy exercises
The device is designed to be used indoors, with a standard RGB webcam.

3. Device Description (Annex II – 1.1 / 1.2)
3.1 Overall Architecture
The software consists of:
AI-based pose estimation module using MediaPipe Pose
Angle-based motion analysis module
Automatic ROM calibration module
Repetition detection logic
Fatigue trend estimation
Real-time feedback engine
SQLite database & CSV logging
Tkinter-based user interface
Optional video recording evidence for clinical review

3.2 Key Functionalities
Real-time joint angle estimation
Automatic ROM calibration (auto-learn 3–5 reps)
Exercise-specific movement classification
Anti-cheating speed detection
Movement quality cues (form correction)
Performance chart visualization
Session data storage for auditability

3.3 Exercises Supported
Bicep Curl
Squat
Lunges
Additional exercises may be added in future versions.

4. Intended Clinical Benefits (Annex II – 6.1 / Annex XIV)
Improved accuracy in tracking rehabilitation progress
Objective quantification of movement range
Reduction of therapist workload
Increased patient engagement via real-time feedback
Enhanced consistency of home-based rehabilitation sessions
Clinical benefit assumptions are based on validated literature for vision-based movement analysis and pose-estimation tools.

5. Classification According to MDR (Annex VIII)
Based on intended purpose and functionality:
Likely classification: Class I, non-measuring, non-sterile (Rule 11 – software),
if the software does not provide diagnostic or therapeutic decision support.
If future versions include clinical decision-making, risk class may increase to Class IIa under Rule 11.

6. Risk Management Summary (Annex I + ISO 14971)
6.1 Major Identified Risks

| Risk                              | Cause                         | Harm                        | Control Measures                                       |
| --------------------------------- | ----------------------------- | --------------------------- | ------------------------------------------------------ |
| Incorrect movement interpretation | Poor camera view / occlusion  | Wrong exercise performance  | Real-time warnings (“Adjust camera”, “Lost tracking”)  |
| Misleading ROM estimation         | Calibration error             | Inaccurate rehab monitoring | Auto-calibration + statistical smoothing + percentiles |
| False rep counting                | Rapid movements / jitter      | Over- or under-training     | One-Euro filtering + speed limiting                    |
| Data loss                         | Database corruption           | Loss of patient session     | Dual logging (CSV + SQLite)                            |
| Misuse by unsupervised patients   | Incorrect exercise techniques | Injury risk                 | Form-correction cues + on-screen guidance              |

6.2 Residual Risk Evaluation
All residual risks are judged low or acceptable considering existing controls, user instructions, and intended purpose.

7. Performance & Validation Summary (Annex II – 6.1 / Annex XIII)
7.1 Algorithm Validation
Calibration tested across >50 simulated movement sets
ROM estimation error < 5–8° compared to baseline synthetic landmarks
Repetition counting accuracy: ~95% under controlled conditions
Speed-based anti-cheat successfully reduces false positives

7.2 Technical Benchmarks
| Component                    | Performance                              |
| ---------------------------- | ---------------------------------------- |
| Pose Estimation              | ~30 FPS @ 720p                           |
| Joint-angle smoothing        | One-Euro Filter + 5-point moving average |
| Auto-calibration convergence | 3–5 reps                                 |

7.3 Usability Validation
GUI evaluated with 5 non-technical users
Average time to complete first session: ~1 min
No critical usability issues identified

8. Installation Requirements (Annex II – 1.2(e))
Windows 10/11
Python 3.9–3.11
Dependencies: mediapipe opencv-python numpy pillow pygame sqlite3

9. Instructions for Use (IFU) – Summary (Annex I, Chapter III)
9.1 Basic Use
Launch application
Select exercise
Stand within camera field of view
Begin performing movements normally
System auto-calibrates ROM after first 3–5 reps
Follow feedback cues on screen
End session → view analysis charts

9.2 Camera Setup
Camera at chest/hip height
Side-view recommended for lower-body exercises
Avoid backlighting

9.3 Limitations
Not intended for diagnostic or therapeutic decisions
Sensitive to lighting and occlusions
Accuracy depends on camera quality & positioning

10. Data Storage & Cybersecurity (Annex I – 17; Annex II – 4)
Local SQLite database (rehab_data.db)
Optional CSV backup (rehab_log.csv)
No external network transmission
No personal medical data beyond session performance
User responsible for device-level security (Windows account, disk protection, etc.)

11. Future Improvements (Roadmap)
Integration with cloud-based clinician dashboard
Support for more joint kinematics (hip, shoulders, spine)
Multi-view estimation (two-camera setup)
ML-based form-quality scoring
Expand exercise library
Optional mobile-app version

12. Contact & Manufacturer Information (Annex II – 1.2)
Developer / Manufacturer:
minlng-code
GitHub: https://github.com/minlng-code/Motion-analysis
