import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

const RiskResult = ({ result, onReset }) => {
    const { risk_label, risk_color, probability, shap_values } = result;

    const getIcon = () => {
        if (risk_label === 'Low Risk') return <CheckCircle className="w-16 h-16 text-green-500" />;
        if (risk_label === 'Medium Risk') return <AlertTriangle className="w-16 h-16 text-yellow-500" />;
        return <XCircle className="w-16 h-16 text-red-500" />;
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card text-center"
        >
            <div className="flex justify-center mb-4">
                {getIcon()}
            </div>

            <h2 className="text-3xl font-bold mb-2" style={{ color: risk_color }}>
                {risk_label}
            </h2>

            <div className="text-xl text-gray-300 mb-8">
                Probability: <span className="font-mono font-bold">{(probability * 100).toFixed(2)}%</span>
            </div>

            <div className="text-left bg-slate-800/50 p-6 rounded-lg mb-8">
                <h3 className="text-lg font-semibold mb-4 border-b border-gray-700 pb-2">Key Factors (SHAP)</h3>
                <div className="space-y-3">
                    {shap_values.slice(0, 5).map((item, idx) => (
                        <div key={idx} className="flex justify-between items-center text-sm">
                            <span className="text-gray-300">{item.feature}</span>
                            <span className={item.shap_value > 0 ? 'text-red-400' : 'text-green-400'}>
                                {item.shap_value > 0 ? '+' : ''}{item.shap_value.toFixed(4)}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            <button onClick={onReset} className="btn bg-gray-600 hover:bg-gray-500 w-full">
                Analyze Another Application
            </button>
        </motion.div>
    );
};

export default RiskResult;
