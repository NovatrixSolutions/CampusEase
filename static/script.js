// API Base URL
const API_URL = 'http://localhost:5000';

// Form submission handler
const form = document.getElementById('predictorForm');
if (form) {
  form.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = {};
    const skills = [];
    let additionalSkillsCount = 0;
    
    // Collect form data
    for (let [key, value] of formData.entries()) {
      if (key === 'Internship_Experience') {
        data[key] = 1;
        skills.push('Internship');
      } else if (key === 'Python_Skill') {
        data[key] = 1;
        skills.push('Python');
      } else if (key === 'DSA_Skill') {
        data[key] = 1;
        skills.push('DSA');
      } else if (key === 'SQL_Skill') {
        data[key] = 1;
        skills.push('SQL');
      } else if (key.endsWith('_Skill')) {
        additionalSkillsCount++;
        skills.push(key.replace('_Skill', ''));
      } else {
        data[key] = parseFloat(value) || 0;
      }
    }
    
    // Set defaults for unchecked
    if (!data.Internship_Experience) data.Internship_Experience = 0;
    if (!data.Python_Skill) data.Python_Skill = 0;
    if (!data.DSA_Skill) data.DSA_Skill = 0;
    if (!data.SQL_Skill) data.SQL_Skill = 0;
    
    // Add Skills field
    data.Skills = skills.join(', ');
    data.Target_Role = '';
    
    // Show loading
    const resultPanel = document.getElementById('resultPanel');
    const resultIdle = document.getElementById('resultIdle');
    const resultContent = document.getElementById('resultContent');
    
    if (resultIdle) resultIdle.style.display = 'none';
    if (resultContent) {
      resultContent.style.display = 'block';
      resultContent.innerHTML = '<p style="text-align:center;">Analyzing...</p>';
    }
    
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }
      
      const result = await response.json();
      displayResult(result);
      
    } catch (error) {
      if (resultContent) {
        resultContent.innerHTML = `<p style="color:red;text-align:center;">Error: ${error.message}</p>`;
      }
    }
  });
}

function displayResult(result) {
  const resultContent = document.getElementById('resultContent');
  if (!resultContent) return;
  
  const probability = result.probability;
  const prediction = result.prediction;
  const recommendations = result.recommendations || [];
  
  let html = `
    <div class="result-score">
      <h2>${probability}</h2>
      <p>Placement Probability</p>
    </div>
    <div class="result-status ${prediction === 'Placed' ? 'placed' : 'not-placed'}">
      ${prediction}
    </div>
  `;
  
  if (recommendations.length > 0) {
    html += '<div class="recommendations"><h4>Recommendations</h4><ul>';
    recommendations.forEach(rec => {
      html += `<li>${rec}</li>`;
    });
    html += '</ul></div>';
  }
  
  resultContent.innerHTML = html;
}
