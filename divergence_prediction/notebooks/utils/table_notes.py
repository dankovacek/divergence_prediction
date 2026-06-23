notes_html = """
<h2>Notes:</h2>
<p style="margin-left:10px;">
<h3>Methods:</h3>
<strong>timeEnsemble</strong>: Temporally-averaged LSTM ensemble.<br><br>
<strong>distEnsemble</strong>: Distribution-averaged LSTM ensemble.<br><br>
<strong>LN<sub>MLE</sub></strong>: Log-normal distribution fitted by maximum likelihood to observed data<br><br>
<strong>LN<sub>PredictedLog</sub></strong>: Log-normal distribution using predicted log-mean \( \mu \) and log-standard deviation \( \sigma \), i.e., parameters of \( \log Q \sim \mathcal{N}(\mu, \sigma^2) \).<br><br>
<strong>LN<sub>PredictedMOM</sub></strong>: Log-normal distribution using predicted mean and standard deviation of \( Q \), with parameters estimated by method of moments.<br><br>
<strong>KNN</strong>: Attribute-based ensemble estimate using average distribution from the 2 and 8 nearest neighbors (2/8-NN) selected by similarity in catchment attributes.<br><br>
<h3>Performance Measures</h3>
<strong>RMSE</strong>: Root Mean Square Error, \( \\sqrt{ \\frac{1}{N} \\sum_{t=1}^{N} \\left( Q_{\\text{sim}}(t) - Q_{\\text{obs}}(t) \\right)^2 } \)<br><br>
<strong>VE</strong>: Volume Efficiency, \( 1 - \\frac{ \\left| \\sum Q_{\\text{sim}} - \\sum Q_{\\text{obs}} \\right| }{ \\sum Q_{\\text{obs}} } \)<br><br>
<strong>PMF/FDC<sub>Volume Bias</sub></strong>: Relative bias in expected flow from estimated PMFs/FDCs, \( \\frac{ \\mathbb{E}[Q_{\\text{sim}}] - \\mathbb{E}[Q_{\\text{obs}}] }{ \\mathbb{E}[Q_{\\text{obs}}] } \)<br><br>
<strong>RE</strong>: Relative error, computed as \( \\frac{F_{\\text{sim}}(p) - F_{\\text{obs}}(p)}{F_{\\text{obs}}(p)} \), where \( F(p) \) denotes the duration curve (FDC) for exceedance duration \(p\).<br><br>
<strong>NSE</strong>: Nash–Sutcliffe Efficiency, defined as \( 1 - \\frac{\\sum (Q_{\\text{sim}} - Q_{\\text{obs}})^2}{\\sum (Q_{\\text{obs}} - \\bar{Q}_{\\text{obs}})^2} \). Values closer to 1 indicate better performance.<br><br>
<strong>KGE</strong>: Kling–Gupta Efficiency, combining correlation, variability, and bias:<br>
\( \\quad \\mathrm{KGE} = 1 -\ \sqrt{(r - 1)^2 + (\\alpha - 1)^2 + (\\beta - 1)^2} \)<br>
\(\\quad\)<strong>r</strong>: Pearson correlation; <strong>α</strong>: ratio of standard deviations \( \\left( \\frac{\\sigma_{\\text{sim}}}{\\sigma_{\\text{obs}}} \\right) \); <strong>β</strong>: bias ratio \( \\left( \\frac{\\mu_{\\text{sim}}}{\\mu_{\\text{obs}}} \\right) \)<br><br>
<strong>KLD</strong>: Kullback–Leibler divergence between observed and simulated distributions:<br>
\( \\quad D_{\\mathrm{KL}}(P \\parallel Q) = \\sum P(x) \\log \\left(\\frac{P(x)}{Q(x)}\\right) \)<br><br>
<strong>EMD</strong>: Sum of absolute differences between cumulative distributions: \( \\mathrm{EMD}(P, Q) = \\sum_x \\left| \\mathrm{CDF}_P(x) - \\mathrm{CDF}_Q(x) \\right| \)
</p>
"""