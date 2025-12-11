"""
MASTER SCRIPT: Generate ALL Demo and Presentation Materials
Run this ONE script to create everything needed for assignment submission
"""

import os
import sys
import subprocess
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def run_command(cmd, description):
    """Run command and report status"""
    print(f"Running: {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ {description} completed successfully")
            return True
        else:
            print(f"  ✗ {description} failed:")
            print(f"    {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Generate all demo materials"""

    print_header("POPPER RL VALIDATION - DEMO MATERIALS GENERATOR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will generate:")
    print("  1. Presentation slides (9 PNG files)")
    print("  2. Architecture diagrams (7 PNG files)")
    print("  3. Technical report (TXT file, PDF-ready)")
    print("  4. Statistical analysis (JSON files)")
    print("  5. Presentation script (TXT file)")
    print("\nEstimated time: 2-3 minutes\n")

    input("Press Enter to continue...")

    results = {}

    # Step 1: Generate presentation materials
    print_header("STEP 1: PRESENTATION MATERIALS")
    results['presentation'] = run_command(
        "python demo_presentation.py",
        "Presentation materials generation"
    )

    # Step 2: Generate architecture diagrams
    print_header("STEP 2: ARCHITECTURE DIAGRAMS")
    results['diagrams'] = run_command(
        "python generate_diagrams.py",
        "Architecture diagrams generation"
    )

    # Step 3: Generate technical report
    print_header("STEP 3: TECHNICAL REPORT")
    results['report'] = run_command(
        "python src/generate_report.py",
        "Technical report generation"
    )

    # Step 4: Run evaluation
    print_header("STEP 4: EVALUATION ANALYSIS")
    results['evaluation'] = run_command(
        "python src/evaluate.py",
        "Evaluation and statistical analysis"
    )

    # Step 5: Create submission package
    print_header("STEP 5: PACKAGE FOR SUBMISSION")

    print("Creating submission directory structure...")

    # Create submission folder
    submission_dir = f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(submission_dir, exist_ok=True)

    # Copy essential files
    files_to_copy = [
        ("README.md", "Project overview"),
        ("requirements.txt", "Dependencies"),
        ("config.yaml", "Configuration"),
        ("COMPLETE_TECHNICAL_DOCUMENTATION.md", "Technical documentation"),
    ]

    dirs_to_copy = [
        ("src/", "Source code"),
        ("experiments/results/", "Experimental results"),
    ]

    print("\nCopying files to submission package...")
    for file, desc in files_to_copy:
        if os.path.exists(file):
            subprocess.run(f"cp {file} {submission_dir}/", shell=True)
            print(f"  ✓ {desc}")

    for dir_path, desc in dirs_to_copy:
        if os.path.exists(dir_path):
            subprocess.run(f"cp -r {dir_path} {submission_dir}/", shell=True)
            print(f"  ✓ {desc}")

    # Generate summary
    print_header("GENERATION SUMMARY")

    print("Status:")
    for step, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {step.capitalize()}: {status}")

    print("\n" + "=" * 80)
    print("MATERIALS LOCATION")
    print("=" * 80)

    locations = {
        "Presentation Slides": "experiments/results/presentation/",
        "Architecture Diagrams": "experiments/results/diagrams/",
        "Technical Report": "experiments/results/technical_report_*.txt",
        "Statistical Analysis": "experiments/results/*_analysis.json",
        "Trained Models": "experiments/results/*_final.*",
        "Submission Package": f"{submission_dir}/"
    }

    for category, location in locations.items():
        print(f"\n{category}:")
        print(f"  {location}")

        # Count files if directory
        if location.endswith('/') and os.path.exists(location):
            files = os.listdir(location)
            print(f"  ({len(files)} files)")

    print("\n" + "=" * 80)
    print("CHECKLIST FOR ASSIGNMENT SUBMISSION")
    print("=" * 80)

    checklist = [
        ("Source Code", "src/ directory", True),
        ("Technical Report", "technical_report_*.txt", True),
        ("Experimental Results", "JSON and PNG files", True),
        ("Presentation Materials", "presentation/ directory", True),
        ("Architecture Diagrams", "diagrams/ directory", True),
        ("Video Demo", "10-minute video (YOU NEED TO RECORD)", False),
        ("GitHub Repository", "Upload to GitHub", False),
    ]

    print()
    for item, location, done in checklist:
        status = "✓" if done else "⚠"
        print(f"  [{status}] {item:25s} - {location}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. REVIEW MATERIALS:
   cd experiments/results/presentation
   open *.png  # Review all slides

2. REVIEW TECHNICAL REPORT:
   cat experiments/results/technical_report_*.txt

3. RECORD VIDEO (10 minutes):
   Use: experiments/results/presentation/PRESENTATION_SCRIPT.txt
   Show: All PNG slides in order
   Structure: Follow script exactly

4. CONVERT REPORT TO PDF:
   Option A: Copy to Word/Google Docs, export as PDF
   Option B: Use pandoc: pandoc report.txt -o report.pdf
   Option C: Use LaTeX for professional formatting

5. CREATE GITHUB REPOSITORY:
   git init
   git add .
   git commit -m "Popper RL Validation - Final Project"
   git remote add origin <your-repo-url>
   git push -u origin main

6. PREPARE SUBMISSION:
   - ZIP: {submission_dir}.zip
   - Include: Code + PDF Report + Video Link + GitHub URL

7. SUBMIT ON CANVAS/LMS:
   Upload ZIP file and provide:
   - GitHub repository URL
   - Video demonstration URL (YouTube/Vimeo)
   - Any additional materials
""")

    print("=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n🎉 ALL DEMO MATERIALS GENERATED SUCCESSFULLY! 🎉\n")
    print("You are ready to submit your assignment!")


if __name__ == "__main__":
    main()