# Source Mapper

A web application that analyzes academic text for citations and rhetorical moves. This tool helps researchers and students understand how sources are used in academic writing.

## Features

- **Citation Detection**: Identifies various citation styles including APA, MLA, and Chicago
- **Rhetorical Move Analysis**: Classifies sentences into reporting, transforming, or evaluating rhetorical moves
- **Visual Representation**: Displays analyzed text with color-coded citations and rhetorical moves
- **Sample Text**: Includes a sample academic text for demonstration purposes

## Citation Styles Supported

- **APA Format**: (Author, Year), (Author et al., Year), etc.
- **MLA Format**: (Author Page), (Author and Author Page), etc.
- **Narrative Citations**: "According to Author..." and similar constructions
- **Implicit References**: Recognizes demonstrative pronouns and author references

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/mkane968/citation-analysis-tool.git
   cd citation-analysis-tool
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Run the setup script (installs all dependencies and downloads required data):
   ```
   python setup.py
   ```
   This will automatically:
   - Install all required packages (Flask, NLTK, scikit-learn, etc.)
   - Download necessary NLTK data

4. Run the application:
   ```
   python app_public.py  # Use this for the version without ML models
   # OR
   python app.py         # Use this if you have the ML models
   ```

5. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. Enter or paste academic text in the input area
2. Click "Analyze Text" to process the text
3. View the results showing sentences with their citation information and rhetorical moves
4. Use the "Try Sample Text" button to see how the tool works with a pre-made example

## Note on Machine Learning Models

The machine learning models used for rhetorical move classification are not included in this repository. The application will fall back to rule-based classification if the models are not available. Email me at megan.kane@shu.edu if you would like more information about working with these models.

## Contributing

Contributions to improve the citation detection patterns or the user interface are welcome. Please submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
