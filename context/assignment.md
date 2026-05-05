# T3.7 — Handwritten Indic Script Recognition
The problem. Write a character on the screen (or upload an image) and get the Unicode
prediction. Useful for educational apps, form digitization, and accessibility.
What you will build. Streamlit app with a drawable canvas (streamlit-drawable-canvas):
draw a character with your mouse →live prediction. Optional “practice mode” shows a target
character and gives feedback.
Skills. Small CNNs, training from scratch, Streamlit canvas component.
Approach. A 3-layer CNN (∼300k params) trained from scratch reaches 97%+ in <5 minutes
on Colab T4. No transfer learning needed.

What you will build
Multi-script router 
Combine UCI Devanagari, uTHCD Tamil,
BanglaLekha and Kannada-MNIST. First
model predicts script, second predicts char-
acter.

Since this is a Tier 2 project, you are expected to do a more thorough evaluation of your method. You should use a larger dataset than the one used in Tier 1, and you should evaluate your method on a wider range of metrics and an in-depth analysis. You should also write a report that is similar in style to the ICVGIP papers, which includes an introduction, related work, methodology, results, and conclusion sections.

What you will hand in
1. GitHub repository. Public, with README.md, requirements.txt, training notebook, app
code.
2. Live demo. Deployed on HuggingFace Spaces (free) or Streamlit Community Cloud (free).
If hosting fails, a recorded ≤2-minute screen demo is acceptable.
3. Technical report (PDF). Sections: Introduction, Data, Method, Results, Limitations,
References. Include an Ablation section where necessary.
4. One-slide pitch. Single PNG/PDF you could post on LinkedIn.

Dataset scale
2000 - 20000

Report 
6-8 pages, ICVGIP style. Include an Ablation section where necessary.

Models
Light fine-tune + ablation + com-
parison