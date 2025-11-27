import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { CreditCard, DollarSign, Briefcase, Calendar, Home, Activity, AlertCircle } from 'lucide-react';
import axios from 'axios';

const RiskForm = ({ onSubmit, isLoading }) => {
    const [defaults, setDefaults] = useState(null);
    const [formData, setFormData] = useState({
        person_age: '',
        person_income: '',
        person_emp_length: '',
        loan_amnt: '',
        loan_int_rate: '',
        loan_percent_income: '',
        cb_person_cred_hist_length: '',
        person_home_ownership: 'RENT',
        loan_intent: 'PERSONAL',
        loan_grade: 'A',
        cb_person_default_on_file: 'N'
    });

    useEffect(() => {
        const fetchDefaults = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:5000/api/defaults');
                setDefaults(response.data);
                setFormData(prev => ({ ...prev, ...response.data }));
            } catch (error) {
                console.error("Error fetching defaults:", error);
            }
        };
        fetchDefaults();
    }, []);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit(formData);
    };

    if (!defaults) return <div className="text-center p-4">Loading defaults...</div>;

    return (
        <motion.form
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card"
            onSubmit={handleSubmit}
        >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <CreditCard className="text-blue-400" />
                Application Details
            </h2>

            <div className="grid-2">
                <div className="input-group">
                    <label className="input-label">Age</label>
                    <div className="relative">
                        <input
                            type="number"
                            name="person_age"
                            value={formData.person_age}
                            onChange={handleChange}
                            className="input-field pl-10"
                            required
                        />
                        <Calendar className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
                    </div>
                </div>

                <div className="input-group">
                    <label className="input-label">Annual Income ($)</label>
                    <div className="relative">
                        <input
                            type="number"
                            name="person_income"
                            value={formData.person_income}
                            onChange={handleChange}
                            className="input-field pl-10"
                            required
                        />
                        <DollarSign className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
                    </div>
                </div>

                <div className="input-group">
                    <label className="input-label">Employment Length (Years)</label>
                    <div className="relative">
                        <input
                            type="number"
                            name="person_emp_length"
                            value={formData.person_emp_length}
                            onChange={handleChange}
                            className="input-field pl-10"
                            step="0.1"
                            required
                        />
                        <Briefcase className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
                    </div>
                </div>

                <div className="input-group">
                    <label className="input-label">Loan Amount ($)</label>
                    <div className="relative">
                        <input
                            type="number"
                            name="loan_amnt"
                            value={formData.loan_amnt}
                            onChange={handleChange}
                            className="input-field pl-10"
                            required
                        />
                        <DollarSign className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
                    </div>
                </div>

                <div className="input-group">
                    <label className="input-label">Interest Rate (%)</label>
                    <div className="relative">
                        <input
                            type="number"
                            name="loan_int_rate"
                            value={formData.loan_int_rate}
                            onChange={handleChange}
                            className="input-field pl-10"
                            step="0.1"
                            required
                        />
                        <Activity className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
                    </div>
                </div>

                <div className="input-group">
                    <label className="input-label">Loan % of Income</label>
                    <input
                        type="number"
                        name="loan_percent_income"
                        value={formData.loan_percent_income}
                        onChange={handleChange}
                        className="input-field"
                        step="0.01"
                        max="1"
                        required
                    />
                </div>

                <div className="input-group">
                    <label className="input-label">Credit History Length (Years)</label>
                    <input
                        type="number"
                        name="cb_person_cred_hist_length"
                        value={formData.cb_person_cred_hist_length}
                        onChange={handleChange}
                        className="input-field"
                        required
                    />
                </div>

                <div className="input-group">
                    <label className="input-label">Home Ownership</label>
                    <select
                        name="person_home_ownership"
                        value={formData.person_home_ownership}
                        onChange={handleChange}
                        className="input-field"
                    >
                        <option value="RENT">RENT</option>
                        <option value="OWN">OWN</option>
                        <option value="MORTGAGE">MORTGAGE</option>
                        <option value="OTHER">OTHER</option>
                    </select>
                </div>

                <div className="input-group">
                    <label className="input-label">Loan Intent</label>
                    <select
                        name="loan_intent"
                        value={formData.loan_intent}
                        onChange={handleChange}
                        className="input-field"
                    >
                        <option value="PERSONAL">PERSONAL</option>
                        <option value="EDUCATION">EDUCATION</option>
                        <option value="MEDICAL">MEDICAL</option>
                        <option value="VENTURE">VENTURE</option>
                        <option value="HOMEIMPROVEMENT">HOME IMPROVEMENT</option>
                        <option value="DEBTCONSOLIDATION">DEBT CONSOLIDATION</option>
                    </select>
                </div>

                <div className="input-group">
                    <label className="input-label">Loan Grade</label>
                    <select
                        name="loan_grade"
                        value={formData.loan_grade}
                        onChange={handleChange}
                        className="input-field"
                    >
                        {['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(g => (
                            <option key={g} value={g}>{g}</option>
                        ))}
                    </select>
                </div>

                <div className="input-group">
                    <label className="input-label">Default on File?</label>
                    <div className="flex gap-4 mt-2">
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="radio"
                                name="cb_person_default_on_file"
                                value="N"
                                checked={formData.cb_person_default_on_file === 'N'}
                                onChange={handleChange}
                            /> No
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="radio"
                                name="cb_person_default_on_file"
                                value="Y"
                                checked={formData.cb_person_default_on_file === 'Y'}
                                onChange={handleChange}
                            /> Yes
                        </label>
                    </div>
                </div>
            </div>

            <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                className="btn w-full mt-8 text-lg"
                disabled={isLoading}
            >
                {isLoading ? 'Analyzing...' : 'Assess Risk'}
            </motion.button>
        </motion.form>
    );
};

export default RiskForm;
