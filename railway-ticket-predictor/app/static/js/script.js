document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            // The form names are like "Train Number", so we need to keep them
            data[key] = value;
        });
        
        // Add other necessary fields with dummy data for now
        // In a real scenario, you might have more inputs for these
        data['Age of Passengers'] = 'Adult';
        data['Booking Channel'] = 'IRCTC Website';
        data['Travel Distance'] = 1500;
        data['Number of Stations'] = 10;
        data['Travel Time'] = 24;
        data['Train Type'] = 'Express';
        data['Seat Availability'] = 100;
        data['Special Considerations'] = 'None';
        data['Holiday or Peak Season'] = 'No';


        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            resultDiv.textContent = `Prediction: ${result.prediction}`;
            resultDiv.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            resultDiv.textContent = 'An error occurred. Please try again.';
            resultDiv.style.display = 'block';
        });
    });
});
