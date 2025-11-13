document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = {};
    
    // Get form data
    for (let [key, value] of formData.entries()) {
        // Convert numeric fields to integers
        if (['age', 'studytime', 'failures', 'G1', 'G2', 'absences', 'Medu', 'Fedu', 
             'famrel', 'health', 'freetime'].includes(key)) {
            data[key] = parseInt(value);
        } else {
            data[key] = value;
        }
    }

    // Add default values for removed fields (to maintain model compatibility)
    data['address'] = 'U';        // Urban (most common)
    data['guardian'] = 'mother';   // Mother (most common)
    data['Mjob'] = 'at_home';     // At home (default)
    data['Fjob'] = 'teacher';     // Teacher (default)
    data['Pstatus'] = 'T';        // Living together (most common)
    data['reason'] = 'course';     // Course preference (default)
    data['activities'] = 'no';     // No activities (default)
    data['schoolsup'] = 'yes';     // Educational support (default)
    data['nursery'] = 'yes';       // Attended nursery (most common)
    data['traveltime'] = 2;        // 15-30 min travel (default)
    data['paid'] = 'no';          // No paid classes (default)
    data['famsup'] = 'no';        // No family support (default)
    data['goout'] = 4;            // High going out (default)
    data['Walc'] = 1;             // Very low weekend alcohol (default)
    data['romantic'] = 'no';       // Not in relationship (default)
    data['Dalc'] = 1;             // Very low workday alcohol (default)

            // Show loading
            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.querySelector('.predict-btn').disabled = true;    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Hide loading
        document.getElementById('loading').style.display = 'none';
        document.querySelector('.predict-btn').disabled = false;

        // Analyze the input data for insights
        const insights = generateInsights(data, result);

        // Show result
        const resultDiv = document.getElementById('result');
        const isPassing = result.pass_fail === 'PASS';
        
        resultDiv.className = `result ${isPassing ? 'pass' : 'fail'}`;
        resultDiv.innerHTML = `
            <h3>${isPassing ? '🎉 PREDICTED: PASS' : '❌ PREDICTED: FAIL'}</h3>
            <div style="margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <p><strong>Final Grade:</strong> ${result.predicted_grade}/20 (${result.percentage})</p>
                <p><strong>Status:</strong> ${isPassing ? 'Student is likely to pass the course' : 'Student may need additional support'}</p>
            </div>
            
            <div style="text-align: left; margin-top: 25px;">
                <h4 style="margin-bottom: 15px; font-size: 1.3em;">📊 Why This Result?</h4>
                ${insights.factors.map(factor => `
                    <div style="margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.2); border-radius: 5px;">
                        <strong>${factor.icon} ${factor.title}:</strong> ${factor.description}
                    </div>
                `).join('')}
            </div>
            
            <div style="text-align: left; margin-top: 20px;">
                <h4 style="margin-bottom: 15px; font-size: 1.3em;">💡 Recommendations</h4>
                ${insights.recommendations.map(rec => `
                    <div style="margin: 8px 0; padding: 8px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.2); border-radius: 5px;">
                        <strong>${rec.icon}</strong> ${rec.text}
                    </div>
                `).join('')}
            </div>
        `;
        resultDiv.style.display = 'block';

    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        document.querySelector('.predict-btn').disabled = false;
        
        const resultDiv = document.getElementById('result');
        resultDiv.className = 'result fail';
        resultDiv.innerHTML = '<h3>❌ Error</h3><p>Failed to get prediction. Please try again.</p>';
        resultDiv.style.display = 'block';
        document.getElementById('placeholder').style.display = 'none';
    }
});

// Generate intelligent insights based on input data and prediction
function generateInsights(inputData, result) {
    const factors = [];
    const recommendations = [];

    // Analyze key factors (limit to 3 most important)
    if (inputData.G1 >= 15 || inputData.G2 >= 15) {
        factors.push({
            icon: "📈",
            title: "Strong Previous Grades",
            description: `Previous grades (G1: ${inputData.G1}, G2: ${inputData.G2}) show consistent good performance`
        });
    } else if (inputData.G1 < 10 || inputData.G2 < 10) {
        factors.push({
            icon: "📉",
            title: "Previous Grade Concerns",
            description: `Previous grades (G1: ${inputData.G1}, G2: ${inputData.G2}) indicate academic challenges`
        });
    }

    // Study time analysis
    if (inputData.studytime >= 3) {
        factors.push({
            icon: "⏰",
            title: "Good Study Habits",
            description: "Regular study time (5+ hours/week) contributes positively to performance"
        });
    }

    // Failure history
    if (inputData.failures > 0) {
        factors.push({
            icon: "⚠️",
            title: "Past Academic Challenges",
            description: `${inputData.failures} previous failure(s) may impact final grade prediction`
        });
    }

    // Attendance
    if (inputData.absences > 10) {
        factors.push({
            icon: "🏫",
            title: "Attendance Issues",
            description: `High number of absences (${inputData.absences}) negatively affects learning`
        });
    } else if (inputData.absences <= 5) {
        factors.push({
            icon: "✅",
            title: "Good Attendance",
            description: "Low absence rate shows commitment to learning"
        });
    }

    // Family education level
    if (inputData.Medu >= 3 || inputData.Fedu >= 3) {
        factors.push({
            icon: "👨‍👩‍👧‍👦",
            title: "Educational Family Background",
            description: "Parents with higher education provide supportive learning environment"
        });
    }

    // Limit to exactly 3 factors for simplicity
    const limitedFactors = factors.slice(0, 3);    // Generate single recommendation (user requested simplicity)
    if (result.pass_fail === "PASS") {
        recommendations.push({
            icon: "🎯",
            text: "Maintain current study habits and attendance patterns for continued success"
        });
    } else {
        recommendations.push({
            icon: "📈",
            text: "Increase study time and seek additional academic support to improve your final grade"
        });
    }

    return { factors: limitedFactors, recommendations };
}