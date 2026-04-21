"""
Knowledge base for MediCare General Hospital, Hyderabad.
10 documents, each covering one specific topic (100-500 words).
"""

from sentence_transformers import SentenceTransformer
import chromadb

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "OPD Timings",
        "text": (
            "MediCare General Hospital OPD (Out-Patient Department) is open Monday to Saturday. "
            "Morning sessions run from 8:00 AM to 1:00 PM. Evening sessions run from 5:00 PM to 8:00 PM. "
            "The hospital is closed on Sundays for OPD services, but the Emergency Department remains open 24/7 "
            "throughout the week including Sundays and public holidays. "
            "Special clinics such as Diabetology, Cardiology, and Neurology are available on specific weekdays. "
            "Please call the helpline at 040-12345678 or check the website to confirm timings for specialist clinics. "
            "Walk-in patients are accepted during OPD hours, but pre-booked appointments are given priority. "
            "Patients are advised to arrive at least 15 minutes before their scheduled slot."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Appointment Booking",
        "text": (
            "Appointments at MediCare General Hospital can be booked through three channels: "
            "by calling the central helpline at 040-12345678, through the hospital website at www.medicarehyd.in, "
            "or by visiting the reception desk in person during OPD hours. "
            "Online booking is available 24/7. Phone booking is available Monday to Saturday from 7:00 AM to 9:00 PM. "
            "When booking, patients need to provide their full name, date of birth, contact number, and preferred doctor. "
            "A confirmation SMS is sent once the appointment is confirmed. "
            "Cancellations must be made at least 2 hours before the appointment time. "
            "Repeated no-shows may result in temporary suspension of online booking privileges. "
            "For urgent same-day appointments, please call the helpline directly."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Doctor Directory and Specialties",
        "text": (
            "MediCare General Hospital has specialists across 15 departments. "
            "Cardiology: Dr. Ramesh Nair (Mon, Wed, Fri mornings). "
            "Orthopaedics: Dr. Sunita Rao (Tue, Thu, Sat mornings). "
            "Neurology: Dr. Praveen Mehta (Mon, Thu evenings). "
            "Gynaecology & Obstetrics: Dr. Anita Sharma (daily mornings). "
            "Paediatrics: Dr. Kiran Reddy (Mon–Sat mornings). "
            "General Medicine: Dr. Suresh Kumar (daily, both sessions). "
            "Dermatology: Dr. Priya Bose (Tue, Fri evenings). "
            "Gastroenterology: Dr. Venkat Rao (Wed, Sat mornings). "
            "For a complete and up-to-date list of doctors, visiting schedules, and qualifications, "
            "please call 040-12345678 or visit the website. Doctor availability is subject to change."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Consultation Fees",
        "text": (
            "Consultation fees at MediCare General Hospital vary by department and seniority of the doctor. "
            "General Medicine: Rs. 300 per consultation. "
            "Specialist consultations (Cardiology, Neurology, Gastroenterology): Rs. 600 per consultation. "
            "Senior Consultant or HOD consultations: Rs. 800 per consultation. "
            "Paediatrics: Rs. 400 per consultation. "
            "Gynaecology: Rs. 500 per consultation. "
            "Follow-up visits within 7 days of the original consultation are charged at 50% of the original fee. "
            "Fees are payable at the billing counter before the consultation. "
            "Accepted payment modes: cash, UPI, debit card, credit card, and net banking. "
            "These fees are subject to revision. Please confirm current fees at the billing counter or helpline."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Insurance and Cashless Treatment",
        "text": (
            "MediCare General Hospital is empanelled with over 25 insurance providers for cashless treatment. "
            "Empanelled insurers include Star Health, United India, New India Assurance, HDFC ERGO, ICICI Lombard, "
            "Bajaj Allianz, Max Bupa, and the Central Government Health Scheme (CGHS). "
            "For cashless admission, patients must present their insurance card and a valid photo ID at the insurance desk. "
            "Pre-authorisation is required for planned procedures and must be obtained at least 48 hours in advance. "
            "Emergency cashless admission is processed within 4 hours. "
            "Co-payment and sub-limit clauses depend on the patient's specific policy — please verify with your insurer. "
            "The insurance helpdesk is located at the ground floor near the main reception and is open from 8 AM to 8 PM. "
            "For reimbursement claims, all original bills and discharge summaries are provided at the time of discharge."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Emergency Services",
        "text": (
            "MediCare General Hospital Emergency Department is operational 24 hours a day, 7 days a week, 365 days a year. "
            "The emergency helpline number is 040-99999999. Ambulance services are available by calling the same number. "
            "The emergency department handles trauma, cardiac emergencies, strokes, poisoning, burns, and all life-threatening conditions. "
            "A triage nurse assesses all walk-in emergency patients immediately upon arrival. "
            "Critical patients are stabilised in the Resuscitation Bay with attending emergency physicians. "
            "The ICU (Intensive Care Unit) and CCU (Cardiac Care Unit) are directly attached to the Emergency Department. "
            "Family members of emergency patients should report to the Emergency Reception desk for updates. "
            "Do NOT delay calling 040-99999999 if you believe a situation is life-threatening."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Pharmacy Services",
        "text": (
            "The in-hospital pharmacy at MediCare General Hospital is located on the ground floor, adjacent to the main OPD block. "
            "The pharmacy is open from 7:00 AM to 10:00 PM on all days including Sundays. "
            "Both prescription and over-the-counter medicines are available. "
            "The pharmacy stocks branded and generic equivalents — the pharmacist can advise on cost-effective generics. "
            "For inpatients, medicines prescribed by the treating doctor are delivered directly to the ward. "
            "A 10% discount on the MRP is offered to patients with confirmed appointments at the hospital. "
            "The pharmacy also stocks surgical supplies, diabetic monitoring equipment, and orthopaedic aids. "
            "A separate billing counter for pharmacy is available to avoid queues at the main billing. "
            "Patients are advised to retain all pharmacy bills for insurance reimbursement."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Laboratory and Diagnostic Services",
        "text": (
            "MediCare Diagnostics, the in-house lab, is NABL accredited and offers over 500 tests. "
            "The lab is open from 6:30 AM to 8:00 PM on weekdays and 6:30 AM to 2:00 PM on Sundays. "
            "Fasting blood samples for tests like lipid profile, blood sugar, and liver function should be collected before 9:00 AM. "
            "Reports for routine blood tests are available within 4–6 hours. "
            "Culture and sensitivity reports take 48–72 hours. "
            "Radiology services including X-ray, ultrasound, CT scan, and MRI are available on the first floor. "
            "Advance booking is required for MRI and CT scans — call 040-12345678 extension 203. "
            "Home sample collection is available within a 10 km radius of the hospital for an additional fee of Rs. 150. "
            "Patients can access their reports online via the patient portal at www.medicarehyd.in/reports."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Health Packages",
        "text": (
            "MediCare General Hospital offers curated preventive health check-up packages for individuals and corporates. "
            "Basic Health Package (Rs. 999): CBC, blood sugar fasting, lipid profile, urine routine, BMI assessment, physician consultation. "
            "Comprehensive Health Package (Rs. 2499): All basic tests plus thyroid profile, kidney function, liver function, ECG, chest X-ray, and doctor consultation. "
            "Senior Citizen Package (Rs. 3499): All comprehensive tests plus bone density, eye check-up, dental screening, and nutritionist consultation. "
            "Corporate packages are available for groups of 10 or more employees at discounted rates. "
            "Health packages are available Monday to Saturday and must be pre-booked at least 24 hours in advance. "
            "Fasting for 10–12 hours is required before attending the health check. "
            "A detailed report with doctor's recommendations is provided within 24 hours of the health check."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Patient Admission and Discharge",
        "text": (
            "Planned admissions at MediCare General Hospital require prior confirmation from the treating doctor. "
            "Patients must report to the Admission Desk on the ground floor with a valid photo ID and insurance card (if applicable). "
            "A security deposit is collected at the time of admission — the amount depends on the type of room selected. "
            "Room categories available: General Ward (Rs. 800/day), Semi-Private Room (Rs. 1800/day), Private Room (Rs. 3000/day), Deluxe Room (Rs. 5000/day). "
            "All rooms include meals, nursing care, and routine monitoring. "
            "The discharge process is initiated by the treating doctor. Discharge summaries, prescriptions, and investigation reports are handed over at discharge. "
            "Final billing is settled at the cashier before leaving. Cashless patients must ensure insurer approval is in place before discharge. "
            "Patients requiring ambulance for transfer to another facility should inform the nursing station in advance."
        ),
    },
]


def build_knowledge_base():
    """Build and return the ChromaDB collection with hospital documents."""
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Building ChromaDB collection...")
    client = chromadb.Client()
    collection = client.create_collection("medicare_kb")

    texts = [doc["text"] for doc in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()
    ids = [doc["id"] for doc in DOCUMENTS]
    metadatas = [{"topic": doc["topic"]} for doc in DOCUMENTS]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"KB built with {len(DOCUMENTS)} documents.")
    return embedder, collection


def retrieval_test(embedder, collection):
    """Quick sanity check — run before building the graph."""
    test_queries = [
        "What are the OPD timings?",
        "How do I book an appointment?",
        "Which insurance companies are accepted?",
        "What is the emergency number?",
        "What health packages are available?",
    ]
    print("\n--- Retrieval Test ---")
    for q in test_queries:
        qe = embedder.encode([q]).tolist()
        results = collection.query(query_embeddings=qe, n_results=1)
        topic = results["metadatas"][0][0]["topic"]
        print(f"Q: {q}\n  → Retrieved: [{topic}]\n")
    print("--- Retrieval Test Complete ---\n")


if __name__ == "__main__":
    embedder, collection = build_knowledge_base()
    retrieval_test(embedder, collection)
