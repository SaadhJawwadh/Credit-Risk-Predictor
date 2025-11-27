import React, { useState } from 'react';
import axios from 'axios';
import RiskForm from './components/RiskForm';
import RiskResult from './components/RiskResult';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async (formData) => {
    setIsLoading(true);
    try {
      // Convert inputs to appropriate types
      const payload = {
        ...formData,
        person_age: Number(formData.person_age),
        person_income: Number(formData.person_income),
        person_emp_length: Number(formData.person_emp_length),
        loan_amnt: Number(formData.loan_amnt),
        loan_int_rate: Number(formData.loan_int_rate),
        loan_percent_income: Number(formData.loan_percent_income),
        cb_person_cred_hist_length: Number(formData.cb_person_cred_hist_length),
      };

      const response = await axios.post('http://127.0.0.1:5000/api/predict', payload);
      setResult(response.data);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to get prediction. Ensure backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Credit Risk AI</h1>
        <p className="text-gray-400">Advanced Machine Learning Assessment</p>
      </header>

      <main className="max-w-2xl mx-auto">
        <AnimatePresence mode="wait">
          {!result ? (
            <RiskForm key="form" onSubmit={handlePredict} isLoading={isLoading} />
          ) : (
            <RiskResult key="result" result={result} onReset={() => setResult(null)} />
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
