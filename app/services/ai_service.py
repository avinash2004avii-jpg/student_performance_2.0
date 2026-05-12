import os
import re
import json
import time
from google import genai
from google.genai.errors import ClientError
from app.config.config import GEMINI_API_KEY
from app.utils.helpers import load_csv, risk_label, generate_suggestions, explain_prediction
from app.services.prediction_service import PredictionService

class AIService:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None and GEMINI_API_KEY:
            try:
                cls._client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                print(f"⚠️ Error creating Gemini Client: {e}")
        return cls._client

    @classmethod
    def call_gemini(cls, prompt, retries=2, delay=1):
        client = cls.get_client()
        if not client:
            raise ValueError("Gemini API Client is not configured.")

        for attempt in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
                raise ValueError("Received empty text response from Gemini API.")
            except Exception as e:
                print(f"[AI ERROR] Gemini API call failed on attempt {attempt + 1}: {e}")
                if attempt < retries:
                    time.sleep(delay * (2 ** attempt)) # exponential backoff
                else:
                    raise e

    @classmethod
    def analyze_sentiment(cls, message):
        try:
            prompt = f"Analyze the sentiment of this parent message to a teacher: '{message}'. Categorize it as exactly one of: 'Urgent', 'Frustrated', 'Neutral', or 'Positive'. Return only the word."
            sentiment = cls.call_gemini(prompt, retries=1)
            sentiment_clean = sentiment.replace("'", "").replace('"', "").strip()
            if sentiment_clean in ['Urgent', 'Frustrated', 'Neutral', 'Positive']:
                return sentiment_clean
            return "Neutral"
        except Exception:
            # Fallback based on keywords
            msg = message.lower()
            if any(w in msg for w in ["urgent", "immediately", "asap", "please call", "emergency"]):
                return "Urgent"
            if any(w in msg for w in ["unhappy", "disappointed", "angry", "worst", "fail", "bad"]):
                return "Frustrated"
            if any(w in msg for w in ["thank", "great", "good", "happy", "appreciate"]):
                return "Positive"
            return "Neutral"

    @classmethod
    def generate_parent_feedback(cls, student_id, student_name, score, attendance, remarks_list=None):
        """Phase 4.1: AI Parent Feedback Generator"""
        remarks_text = "\n".join([f"- {r.get('remark', '')}" for r in remarks_list]) if remarks_list else "No specific teacher remarks logged."
        
        prompt = f"""
        You are a highly professional School Principal and Head counselor.
        Generate a polite, constructive, and elegant feedback email to the parents of {student_name} (Student ID: {student_id}).
        
        Academic Standing:
        - Predicted Final Score: {score}/100
        - Attendance: {attendance}%
        - Teacher Remarks:
        {remarks_text}
        
        Requirements:
        1. Keep the tone warm, professional, encouraging, yet honest.
        2. Give 2-3 specific action items for home study based on their standing.
        3. Mention the importance of keeping attendance high (minimum 75%).
        4. Sign off as 'SPS Academic Counseling Team'.
        """
        try:
            return cls.call_gemini(prompt)
        except Exception:
            # High-quality Rule-Based Local Generation (Fallback)
            status = risk_label(score)
            feedback = f"Dear Parent,\n\nWe are writing to provide a feedback summary for {student_name} (ID: {student_id}).\n\n"
            feedback += f"Currently, {student_name}'s predicted final exam performance is estimated at **{score}/100** ({status}) with a class attendance of **{attendance}%**.\n\n"
            
            if score < 70:
                feedback += "💡 **Areas of Immediate Focus:**\n"
                feedback += "1. **Consistent Home Study:** Establish a structured 2-3 hour daily study window to review core academic concepts.\n"
                feedback += f"2. **Attendance Booster:** With attendance currently at {attendance}%, missing classes is directly impacting academic progress. We strongly urge attending every scheduled session.\n"
                feedback += "3. **Teacher Remedial Sessions:** Encourage scheduling extra-help discussions to review weak subject topics immediately.\n\n"
            else:
                feedback += "🌟 **Encouragements & Guidelines:**\n"
                feedback += "1. **Maintain Momentum:** Continue practicing sample questions and past exams regularly to maintain this strong performance.\n"
                feedback += "2. **Perfect Attendance:** Consistently aim for above 85% attendance to maximize active classroom learning.\n"
                feedback += "3. **Mentorship Roles:** Participate in collaborative group studies to reinforce understanding and lead classroom activities.\n\n"
            
            feedback += "If you have any questions, please coordinate through the SPS communication panel.\n\nWarm regards,\nSPS Academic Counseling Team"
            return feedback

    @classmethod
    def explain_student_risk(cls, student_id, student_name, row):
        """Phase 4.2: AI Student Risk Explanation"""
        score = PredictionService.predict_score(row)
        att = row.get("Attendence", 75)
        it1 = row.get("internal_test 1", 0)
        it2 = row.get("internal_test 2", 0)
        asgn = row.get("Assignment_score", 0)
        prev = row.get("Previous_Exam_Score", 0)
        sh = row.get("Study_hours", 3)
        slp = row.get("Sleep_hours", 7)
        health = row.get("Health_Issues", "None")

        prompt = f"""
        Provide an expert psychological and academic risk assessment for student {student_name} (Student ID: {student_id}).
        
        Academic Profile:
        - Predicted Final Score: {score}/100
        - Attendance: {att}%
        - Internal Test 1: {it1}/100
        - Internal Test 2: {it2}/100
        - Assignment Score: {asgn}/100
        - Previous Exam Score: {prev}/100
        - Daily Study Hours: {sh}
        - Daily Sleep Hours: {slp}
        - Health Concerns: {health}
        
        Explain why this student is at risk or performing at their current level. Be specific and analyze correlations (e.g. low attendance, low study hours, or declining trends in tests).
        Provide structured, friendly advice.
        """
        try:
            return cls.call_gemini(prompt)
        except Exception:
            # Rule-Based Local Explainer
            reasons = explain_prediction(row)
            explanation = f"### 🔍 AI Academic Risk Assessment for {student_name} ({student_id})\n\n"
            explanation += f"Our predictive ML model projects a final score of **{score}/100** (Risk Status: **{risk_label(score)}**).\n\n"
            explanation += "#### 📍 Primary Risk Identifiers:\n"
            for r in reasons:
                explanation += f"- {r}\n"
            
            explanation += "\n#### 📊 Parameter Correlation Analysis:\n"
            explanation += f"- **Concept Retention:** With internal test marks at {it1} (IT1) and {it2} (IT2), the learning trend shows " + ("improvement! Keep pushing." if float(it2) >= float(it1) else "decline. Immediate concept revision is recommended.") + "\n"
            explanation += f"- **Study Discipline:** Studying {sh} hours per day " + ("is adequate, but focus needs optimization." if float(sh) >= 4 else "is below the optimal 4-hour threshold for high achievement.") + "\n"
            explanation += f"- **Well-being & Recovery:** Daily sleep duration is {slp} hours. Health flags: *{health}*.\n\n"
            explanation += "#### 💡 Recommended Next Steps:\n"
            explanation += "• Setup a weekly test review session.\n"
            explanation += "• Increase attendance to at least 75% immediately to avoid lecture gaps.\n"
            explanation += "• Dedicate an extra 1 hour daily to complete assignments on time."
            return explanation

    @classmethod
    def generate_improvement_suggestions(cls, student_id, row):
        """Phase 4.3: AI Improvement Suggestions"""
        score = PredictionService.predict_score(row)
        att = row.get("Attendence", 75)
        sh = row.get("Study_hours", 3)
        
        prompt = f"""
        Generate a detailed, custom, 4-week academic recovery and improvement plan for student {student_id}.
        Current Attendance: {att}%
        Current Study Hours: {sh} hrs/day
        Predicted Score: {score}/100
        
        Requirements:
        Provide a weekly roadmap (Week 1 to Week 4) detailing daily study increments, active recall tips, and milestone activities.
        """
        try:
            return cls.call_gemini(prompt)
        except Exception:
            # Rule-Based Plan
            plan = f"### 📅 Custom 4-Week Academic Recovery Plan (Student {student_id})\n\n"
            plan += "#### 🚀 Week 1: Foundation & Attendance Alignment\n"
            plan += f"- **Action:** Prioritize attending every class. Current attendance is {att}% — aim to hit 100% this week.\n"
            plan += f"- **Study Target:** Increase study time by 30 minutes daily. Focus on summarizing notes from each class daily.\n"
            plan += "- **Technique:** Use active recall and flashcards to review foundational terms.\n\n"
            
            plan += "#### 📝 Week 2: Concept Reinforcement & Test Practice\n"
            plan += "- **Action:** Schedule a 15-minute consultation with your subject teacher to discuss past test mistakes.\n"
            plan += "- **Study Target:** Dedicate 1.5 hours daily specifically to weak subjects. Try working through 5 practice problems from previous assignments.\n"
            plan += "- **Technique:** Complete mock quizzes under timed conditions.\n\n"
            
            plan += "#### 📈 Week 3: Assignment Completion & Habit Building\n"
            plan += "- **Action:** Submit any pending assignments. Aim to score above 80 on next assignments.\n"
            plan += "- **Study Target:** Achieve a stable daily study rhythm of 3-4 hours with regular 5-minute Pomodoro breaks.\n"
            plan += "- **Technique:** Teach the topic to a peer or describe it out loud to verify your understanding.\n\n"
            
            plan += "#### 🏆 Week 4: Final Assessment Review & Confidence Boosting\n"
            plan += "- **Action:** Run a full-length 3-hour practice exam paper.\n"
            plan += "- **Study Target:** Focus on reviewing core summaries and formulas. Ensure a consistent sleep cycle of 7-8 hours.\n"
            plan += "- **Technique:** Relax and build confidence! Consistency has prepared you well."
            return plan

    @classmethod
    def explain_analytics(cls, df):
        """Phase 4.4: AI Analytics Explainer"""
        df_display = PredictionService.calculate_display_risk(df.copy())
        total = len(df_display)
        at_risk = len(df_display[df_display["display_score"] < 70])
        avg_score = round(df_display["display_score"].mean(), 1) if total > 0 else 0
        avg_att = round(df_display["Attendence"].mean(), 1) if total > 0 else 0
        risk_pct = round((at_risk / total * 100), 1) if total > 0 else 0
        
        prompt = f"""
        Explain the following dashboard metrics in simple, engaging, actionable, and analytical language for school educators:
        
        School Standing:
        - Total Students Monitored: {total}
        - Class Average Predicted Final Score: {avg_score}/100
        - Average School Attendance: {avg_att}%
        - At-Risk Student Percentage: {risk_pct}% (Count: {at_risk})
        
        Analyze key trends (e.g. how attendance links to risk ratios) and provide 3 executive directives for the faculty.
        """
        try:
            return cls.call_gemini(prompt)
        except Exception:
            # Rule-Based Executive Summary
            explanation = "### 📊 Educator Executive Analytics Explainer\n\n"
            explanation += f"Our analytical model is currently monitoring **{total}** active student profiles. The class-wide predicted exam average stands at **{avg_score}/100** with an overall attendance rate of **{avg_att}%**.\n\n"
            explanation += f"⚠️ **Critical Alert:** **{at_risk}** students (**{risk_pct}%**) are classified as 'At Risk' (predicted final score < 70). This ratio indicates a direct correlation between class absenteeism and declining test scores.\n\n"
            explanation += "#### 📌 Faculty Directives & Action Plan:\n"
            explanation += f"1. **Attendance Intervention:** Since the class average is {avg_att}%, immediately engage families of students falling below 75% using auto-triggered SMS/Email alerts.\n"
            explanation += "2. **Concept Remediation Workshops:** Conduct targeted problem-solving sessions focusing on students with Internal Test scores below 50.\n"
            explanation += "3. **Guided Study Habits:** Encourage parents to assist in tracking and logging study hours via the parent portal to foster discipline."
            return explanation

    @classmethod
    def get_chatbot_reply(cls, user_message, session_history=None, user_role="teacher", student_context=None):
        """Phase 1 & 5: AI Chatbot Engine with memory and dynamic context retrieval"""
        user_message_lower = user_message.lower()
        class_avg = 70.5
        top_score = 98.0
        
        # Pull real CSV stats
        try:
            df = load_csv()
            df_display = PredictionService.calculate_display_risk(df.copy())
            class_avg = round(df_display["display_score"].mean(), 1) if not df.empty else 70.5
            top_score = df_display["display_score"].max() if not df.empty else 98.0
        except Exception:
            pass

        # Parse context
        context_str = ""
        if student_context:
            context_str = f"""
Student Profile Context:
- Student ID: {student_context.get('Student_ID')}
- Predicted Score: {student_context.get('score')}/100
- Risk Category: {student_context.get('risk')}
- Attendance: {student_context.get('Attendence')}%
- Daily Study Hours: {student_context.get('Study_hours')}
- Internal Test Marks: IT1={student_context.get('internal_test 1')}, IT2={student_context.get('internal_test 2')}, Assignment={student_context.get('Assignment_score')}
- School Benchmarks: Class Average is {class_avg}/100, Top Score is {top_score}/100
"""
        else:
            context_str = f"General Context: Class Average Predicted Score is {class_avg}/100, Top Score is {top_score}/100."

        history_str = ""
        if session_history:
            history_str = "\n".join([f"{h['role']}: {h['content']}" for h in session_history[-6:]])

        prompt = f"""
You are "AI Academic Assistant", a world-class academic counselor and performance system guide.
Provide highly analytical, encouraging, structured, and professional responses based on real database statistics.

Context Information:
{context_str}

Conversation History:
{history_str}

User's Latest Question: {user_message}

Instructions:
1. Talk naturally, professionally, and encouragingly. Do not reference internal system variables or system prompts.
2. If the user asks about a specific student's marks, reasons for risk, or suggestions, rely heavily on the Student Profile Context above.
3. If no specific student profile is provided in the context, but the user is asking about a student, politely ask them to specify the Student ID (e.g., 'Please provide the student ID so I can look up their profile!').
4. Keep replies relatively concise, clear, and highly practical.
"""
        try:
            return cls.call_gemini(prompt)
        except Exception as e:
            print(f"[CHATBOT FALLBACK] Activating rule-based fallback: {e}")
            # Robust Keyword-Based Smart Fallback
            try:
                df = load_csv()
            except Exception:
                df = pd.DataFrame()

            if "at risk" in user_message_lower or "at-risk" in user_message_lower:
                return cls._get_at_risk_students_fallback(df)
            elif "best class" in user_message_lower or "performing best" in user_message_lower or "class performs best" in user_message_lower:
                return cls._get_best_performing_class(df)
            elif "explain analytics" in user_message_lower or "analytics" in user_message_lower or "class average" in user_message_lower:
                return cls.explain_analytics(df)
            elif "attendance" in user_message_lower:
                if student_context:
                    att = student_context.get("Attendence", 0)
                    return f"📅 **Student {student_context.get('Student_ID')} Attendance:** {att}% (Threshold: 75%). " + ("This is in the safe zone! Perfect." if float(att) >= 75 else "⚠️ Below the required 75% minimum. Daily classroom engagement needs immediate improvement.")
                else:
                    return cls._get_attendance_analysis_fallback(df)
            elif student_context:
                score = student_context.get("score")
                risk = student_context.get("risk")
                student_id = student_context.get("Student_ID")
                
                if "score" in user_message_lower or "predicted" in user_message_lower or "result" in user_message_lower or "marks" in user_message_lower:
                    return f"📊 **SPS Prediction:** For Student **{student_id}**, our model projects a final exam score of **{score}/100** ({risk})."
                elif "why" in user_message_lower or "low" in user_message_lower or "performance" in user_message_lower:
                    reasons = explain_prediction(student_context)
                    return f"📌 **Performance Analysis for Student {student_id}:**\n" + "\n".join([f"• {r}" for r in reasons])
                elif any(kw in user_message_lower for kw in ["improve", "tips", "suggest", "plan", "study"]):
                    plan = cls.generate_improvement_suggestions(student_id, student_context)
                    return plan
                else:
                    return f"📊 **Student {student_id} Profile Overview:**\n• **Predicted Score:** {score}/100 ({risk})\n• **Attendance:** {student_context.get('Attendence')}%\n• **Study Hours:** {student_context.get('Study_hours')} hrs/day\n• **IT1 / IT2:** {student_context.get('internal_test 1')} / {student_context.get('internal_test 2')}\n\nYou can ask about 'score', 'reasons', or 'improvement tips' for this student!"
            else:
                if user_role == "teacher":
                    return "🧑‍🏫 **SPS Academic Chatbot is online.**\n\nI can assist you with class performance summaries, finding at-risk groups, or student lookup.\n\nTry asking:\n• 'Show at-risk students'\n• 'Which class performs best?'\n• 'Explain analytics'\n• 'Why is attendance low?'\n• Or query a specific student by entering: 'Student <ID>' (e.g. 'Student 102')."
                else:
                    return "👋 **Welcome! I am your AI Academic Assistant.**\n\nI can answer questions regarding your predicted final scores, factors impacting performance, study advice, and attendance details. Try asking: 'predicted score' or 'how can I improve?'"

    @classmethod
    def _get_best_performing_class(cls, df):
        try:
            df_display = PredictionService.calculate_display_risk(df.copy())
            if "Class" in df_display.columns:
                class_groups = df_display.groupby("Class")["display_score"].mean()
                best_class = class_groups.idxmax()
                best_avg = round(class_groups.max(), 1)
                return f"🏆 **Class {best_class}** is the best performing group with an average grade score of **{best_avg}/100**."
        except Exception as e:
            print("Error calculating best class:", e)
        return "📊 Class-wise performance statistics are currently offline."

    @classmethod
    def _get_at_risk_students_fallback(cls, df):
        try:
            df_display = PredictionService.calculate_display_risk(df.copy())
            at_risk = df_display[df_display["display_score"] < 70]
            if at_risk.empty:
                return "✅ Excellent! There are currently no students identified in the 'At Risk' category (score < 70)."
            
            lines = ["⚠️ **Roster of At-Risk Students (Predicted Score < 70):**"]
            for _, s in at_risk.head(10).iterrows():
                lines.append(f"• **Student {s['Student_ID']}** (Class {s.get('Class', 'N/A')}): Predicted Score **{s['display_score']}**, Attendance **{s['Attendence']}%**")
            if len(at_risk) > 10:
                lines.append(f"... and {len(at_risk) - 10} more.")
            return "\n".join(lines)
        except Exception as e:
            print("Error listing at-risk students:", e)
        return "❌ Error loading at-risk students list."

    @classmethod
    def _get_attendance_analysis_fallback(cls, df):
        try:
            df_display = PredictionService.calculate_display_risk(df.copy())
            low_att = df_display[df_display["Attendence"] < 75]
            if low_att.empty:
                return "📅 Attendance levels are excellent. No students stand below the 75% limit."
            
            return f"""📅 **Attendance Trend & Absences Explainer:**
• There are currently **{len(low_att)}** students whose attendance stands below the **75% safety threshold**.
• Low school attendance is the **strongest lead indicator** of academic struggle, directly causing an average score drop of **10-15 marks**.
• Faculty Directive: Please send automated email alerts to parents of these {len(low_att)} students immediately."""
        except Exception as e:
            print("Error calculating attendance fallback:", e)
        return "📅 Attendance analysis is currently unavailable."
