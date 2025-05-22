@echo off
call conda activate fraudenv
streamlit run credit_card_fraud_app.py
pause
