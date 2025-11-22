"""
AI-Powered Patent Claim Strength Analysis

Uses Anthropic Claude to analyze patent claims and assess:
- Claim breadth (narrow/moderate/broad)
- Litigation outcome probability
- Patent vulnerabilities
- Overall strength score (0-100%)
- Comparison to similar litigated patents
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClaimAnalysisResult:
    """Result of AI patent claim analysis."""

    patent_number: str
    drug_name: str

    # Claim characteristics
    claim_breadth: str  # narrow, moderate, broad
    claim_breadth_score: float  # 0-100
    independent_claims_count: int
    dependent_claims_count: int

    # Strength assessment
    overall_strength_score: float  # 0-100
    litigation_probability: float  # 0-100, likelihood patent will be challenged
    upheld_probability: float  # 0-100, likelihood patent survives challenge

    # Vulnerabilities
    vulnerabilities: List[str]
    vulnerability_severity: str  # low, medium, high, critical

    # Prior art concerns
    prior_art_risk: str  # low, medium, high
    prior_art_concerns: List[str]

    # Comparison to similar patents
    similar_patents: List[Dict[str, Any]]
    benchmark_percentile: float  # Where this patent ranks vs similar ones

    # Detailed analysis
    key_claims_summary: str
    novelty_assessment: str
    obviousness_assessment: str
    enablement_assessment: str

    # Recommendations
    recommendations: List[str]

    # Metadata
    analysis_date: date = field(default_factory=date.today)
    model_version: str = "claude-sonnet-4-5-20250929"
    confidence_level: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patent_number": self.patent_number,
            "drug_name": self.drug_name,
            "claim_breadth": self.claim_breadth,
            "claim_breadth_score": self.claim_breadth_score,
            "independent_claims_count": self.independent_claims_count,
            "dependent_claims_count": self.dependent_claims_count,
            "overall_strength_score": self.overall_strength_score,
            "litigation_probability": self.litigation_probability,
            "upheld_probability": self.upheld_probability,
            "vulnerabilities": self.vulnerabilities,
            "vulnerability_severity": self.vulnerability_severity,
            "prior_art_risk": self.prior_art_risk,
            "prior_art_concerns": self.prior_art_concerns,
            "similar_patents": self.similar_patents,
            "benchmark_percentile": self.benchmark_percentile,
            "key_claims_summary": self.key_claims_summary,
            "novelty_assessment": self.novelty_assessment,
            "obviousness_assessment": self.obviousness_assessment,
            "enablement_assessment": self.enablement_assessment,
            "recommendations": self.recommendations,
            "analysis_date": self.analysis_date.isoformat(),
            "model_version": self.model_version,
            "confidence_level": self.confidence_level,
        }


class PatentClaimAnalyzer:
    """
    AI-powered patent claim strength analyzer using Anthropic Claude.

    Analyzes patent claims to assess strength, vulnerabilities, and
    predict litigation outcomes based on claim language and structure.
    """

    # Known Hatch-Waxman litigation outcomes for comparison
    LITIGATION_BENCHMARKS = {
        "adalimumab": {"upheld": False, "outcome": "SETTLED", "strength": 65},
        "lenalidomide": {"upheld": False, "outcome": "SETTLED", "strength": 55},
        "pregabalin": {"upheld": False, "outcome": "INVALIDATED", "strength": 40},
        "dimethyl_fumarate": {"upheld": False, "outcome": "INVALIDATED", "strength": 35},
        "apixaban": {"upheld": True, "outcome": "UPHELD", "strength": 75},
        "rivaroxaban": {"upheld": False, "outcome": "SETTLED", "strength": 60},
        "palbociclib": {"upheld": False, "outcome": "SETTLED", "strength": 58},
        "enzalutamide": {"upheld": True, "outcome": "UPHELD", "strength": 80},
        "ibrutinib": {"upheld": True, "outcome": "ONGOING", "strength": 72},
        "ustekinumab": {"upheld": False, "outcome": "SETTLED", "strength": 62},
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the patent analyzer.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

        if not self.api_key:
            logger.warning(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter. Will use mock analysis for testing."
            )

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None and self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                raise
        return self._client

    def _build_analysis_prompt(
        self,
        patent_number: str,
        drug_name: str,
        claims_text: str,
        patent_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the analysis prompt for Claude.

        Args:
            patent_number: US patent number.
            drug_name: Name of the drug covered by patent.
            claims_text: Full text of patent claims.
            patent_metadata: Additional patent information.

        Returns:
            Structured prompt for analysis.
        """
        metadata_section = ""
        if patent_metadata:
            metadata_section = f"""
Patent Metadata:
- Filing Date: {patent_metadata.get('filing_date', 'Unknown')}
- Issue Date: {patent_metadata.get('issue_date', 'Unknown')}
- Expiration Date: {patent_metadata.get('expiration_date', 'Unknown')}
- Patent Type: {patent_metadata.get('patent_type', 'Unknown')}
- Therapeutic Area: {patent_metadata.get('therapeutic_area', 'Unknown')}
- Assignee: {patent_metadata.get('assignee', 'Unknown')}
"""

        prompt = f"""You are an expert pharmaceutical patent attorney analyzing patent claims for litigation risk assessment in Hatch-Waxman (ANDA) challenges.

Analyze the following patent and provide a detailed assessment:

Patent Number: {patent_number}
Drug Name: {drug_name}
{metadata_section}

PATENT CLAIMS:
{claims_text}

Provide your analysis in the following JSON format:

{{
    "claim_breadth": "narrow|moderate|broad",
    "claim_breadth_score": <0-100>,
    "claim_breadth_reasoning": "<explanation>",

    "independent_claims_count": <number>,
    "dependent_claims_count": <number>,

    "overall_strength_score": <0-100>,
    "strength_reasoning": "<explanation>",

    "litigation_probability": <0-100>,
    "litigation_reasoning": "<why this patent is likely/unlikely to be challenged>",

    "upheld_probability": <0-100>,
    "upheld_reasoning": "<why patent would/wouldn't survive challenge>",

    "vulnerabilities": [
        "<vulnerability 1>",
        "<vulnerability 2>"
    ],
    "vulnerability_severity": "low|medium|high|critical",

    "prior_art_risk": "low|medium|high",
    "prior_art_concerns": [
        "<concern 1>",
        "<concern 2>"
    ],

    "key_claims_summary": "<summary of the key protective claims>",
    "novelty_assessment": "<assessment of novelty under 35 USC 102>",
    "obviousness_assessment": "<assessment of obviousness under 35 USC 103>",
    "enablement_assessment": "<assessment of enablement under 35 USC 112>",

    "similar_litigated_patents": [
        {{
            "drug": "<drug name>",
            "outcome": "UPHELD|INVALIDATED|SETTLED",
            "relevance": "<why similar>"
        }}
    ],

    "recommendations": [
        "<recommendation 1>",
        "<recommendation 2>"
    ],

    "confidence_level": "low|medium|high"
}}

Consider:
1. Breadth of claims - narrow claims are harder to design around but easier to invalidate
2. Prior art risks - especially pre-filing publications or patents
3. Enablement/written description issues
4. Obviousness based on prior compounds in the same class
5. Historical outcomes of similar pharmaceutical patent challenges
6. Specific claim language that creates vulnerabilities (e.g., "consisting of" vs "comprising")

Respond ONLY with the JSON object, no additional text."""

        return prompt

    def _parse_analysis_response(
        self,
        response_text: str,
        patent_number: str,
        drug_name: str,
    ) -> ClaimAnalysisResult:
        """
        Parse Claude's response into structured result.

        Args:
            response_text: Raw response from Claude.
            patent_number: Patent being analyzed.
            drug_name: Drug name.

        Returns:
            Structured ClaimAnalysisResult.
        """
        try:
            # Extract JSON from response (handle potential markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            # Calculate benchmark percentile based on similar patents
            similar_strengths = [
                p.get("strength", 50)
                for p in self.LITIGATION_BENCHMARKS.values()
            ]
            strength = data.get("overall_strength_score", 50)
            below_count = sum(1 for s in similar_strengths if s < strength)
            benchmark_percentile = (below_count / len(similar_strengths)) * 100

            return ClaimAnalysisResult(
                patent_number=patent_number,
                drug_name=drug_name,
                claim_breadth=data.get("claim_breadth", "moderate"),
                claim_breadth_score=float(data.get("claim_breadth_score", 50)),
                independent_claims_count=int(data.get("independent_claims_count", 0)),
                dependent_claims_count=int(data.get("dependent_claims_count", 0)),
                overall_strength_score=float(data.get("overall_strength_score", 50)),
                litigation_probability=float(data.get("litigation_probability", 50)),
                upheld_probability=float(data.get("upheld_probability", 50)),
                vulnerabilities=data.get("vulnerabilities", []),
                vulnerability_severity=data.get("vulnerability_severity", "medium"),
                prior_art_risk=data.get("prior_art_risk", "medium"),
                prior_art_concerns=data.get("prior_art_concerns", []),
                similar_patents=data.get("similar_litigated_patents", []),
                benchmark_percentile=benchmark_percentile,
                key_claims_summary=data.get("key_claims_summary", ""),
                novelty_assessment=data.get("novelty_assessment", ""),
                obviousness_assessment=data.get("obviousness_assessment", ""),
                enablement_assessment=data.get("enablement_assessment", ""),
                recommendations=data.get("recommendations", []),
                confidence_level=data.get("confidence_level", "medium"),
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Return a default result with error indication
            return self._create_default_result(patent_number, drug_name, error=str(e))

    def _create_default_result(
        self,
        patent_number: str,
        drug_name: str,
        error: Optional[str] = None,
    ) -> ClaimAnalysisResult:
        """Create a default/mock result for testing or error cases."""
        return ClaimAnalysisResult(
            patent_number=patent_number,
            drug_name=drug_name,
            claim_breadth="moderate",
            claim_breadth_score=50.0,
            independent_claims_count=3,
            dependent_claims_count=15,
            overall_strength_score=55.0,
            litigation_probability=70.0,
            upheld_probability=50.0,
            vulnerabilities=[
                "Claims may be overly broad",
                "Potential prior art in related compounds",
            ] if not error else [f"Analysis error: {error}"],
            vulnerability_severity="medium",
            prior_art_risk="medium",
            prior_art_concerns=["Similar compounds disclosed in earlier patents"],
            similar_patents=[],
            benchmark_percentile=50.0,
            key_claims_summary="Mock analysis - API key required for real analysis",
            novelty_assessment="Requires detailed analysis",
            obviousness_assessment="Requires detailed analysis",
            enablement_assessment="Requires detailed analysis",
            recommendations=["Obtain full AI analysis with valid API key"],
            confidence_level="low",
        )

    def analyze_patent_claims(
        self,
        patent_number: str,
        drug_name: str,
        claims_text: str,
        patent_metadata: Optional[Dict[str, Any]] = None,
    ) -> ClaimAnalysisResult:
        """
        Analyze patent claims using Claude AI.

        Args:
            patent_number: US patent number (e.g., "6090382").
            drug_name: Name of the drug covered by the patent.
            claims_text: Full text of the patent claims.
            patent_metadata: Optional dict with filing_date, issue_date, etc.

        Returns:
            ClaimAnalysisResult with comprehensive analysis.
        """
        logger.info(f"Analyzing patent {patent_number} for {drug_name}")

        # If no API key, return mock result
        if not self.api_key:
            logger.warning("Using mock analysis - no API key configured")
            return self._create_mock_analysis(patent_number, drug_name, claims_text)

        # Build prompt
        prompt = self._build_analysis_prompt(
            patent_number=patent_number,
            drug_name=drug_name,
            claims_text=claims_text,
            patent_metadata=patent_metadata,
        )

        try:
            # Call Claude API
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text

            # Parse response
            result = self._parse_analysis_response(
                response_text=response_text,
                patent_number=patent_number,
                drug_name=drug_name,
            )

            logger.info(
                f"Analysis complete for {patent_number}: "
                f"strength={result.overall_strength_score:.1f}%, "
                f"upheld_prob={result.upheld_probability:.1f}%"
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing patent {patent_number}: {e}")
            return self._create_default_result(patent_number, drug_name, error=str(e))

    def _create_mock_analysis(
        self,
        patent_number: str,
        drug_name: str,
        claims_text: str,
    ) -> ClaimAnalysisResult:
        """
        Create a realistic mock analysis based on claims characteristics.

        Used when API key is not available or for testing.
        """
        # Count claims (rough heuristic)
        independent_count = len(re.findall(r'^\s*\d+\.\s+[A-Z]', claims_text, re.MULTILINE))
        dependent_count = len(re.findall(r'claim\s+\d+', claims_text, re.IGNORECASE))

        if independent_count == 0:
            independent_count = 3
        if dependent_count == 0:
            dependent_count = 15

        # Determine breadth heuristically
        broad_indicators = ["comprising", "including", "one or more", "at least"]
        narrow_indicators = ["consisting of", "consisting essentially of", "exactly"]

        broad_count = sum(claims_text.lower().count(ind) for ind in broad_indicators)
        narrow_count = sum(claims_text.lower().count(ind) for ind in narrow_indicators)

        if broad_count > narrow_count * 2:
            claim_breadth = "broad"
            breadth_score = 75.0
        elif narrow_count > broad_count:
            claim_breadth = "narrow"
            breadth_score = 35.0
        else:
            claim_breadth = "moderate"
            breadth_score = 55.0

        # Check for known drug benchmarks
        drug_lower = drug_name.lower().replace(" ", "_")
        benchmark = self.LITIGATION_BENCHMARKS.get(drug_lower, {})

        if benchmark:
            strength = benchmark.get("strength", 55)
            upheld = 70.0 if benchmark.get("upheld", False) else 40.0
        else:
            # Estimate based on claim characteristics
            strength = min(90, max(30, 50 + (breadth_score - 50) * 0.3 + independent_count * 2))
            upheld = strength * 0.9

        vulnerabilities = []
        if claim_breadth == "broad":
            vulnerabilities.append("Broad claim scope may be vulnerable to prior art challenges")
        if "method" in claims_text.lower():
            vulnerabilities.append("Method claims may be harder to enforce than composition claims")
        if "pharmaceutical" in claims_text.lower() and "formulation" in claims_text.lower():
            vulnerabilities.append("Formulation patents often face design-around challenges")
        if not vulnerabilities:
            vulnerabilities.append("Standard pharmaceutical patent vulnerabilities may apply")

        # Determine vulnerability severity
        if len(vulnerabilities) >= 3:
            severity = "high"
        elif len(vulnerabilities) >= 2:
            severity = "medium"
        else:
            severity = "low"

        # Similar patents from benchmarks
        similar = []
        for drug, data in list(self.LITIGATION_BENCHMARKS.items())[:3]:
            similar.append({
                "drug": drug.replace("_", " ").title(),
                "outcome": data["outcome"],
                "relevance": f"Similar pharmaceutical composition patent with {data['outcome'].lower()} outcome"
            })

        # Calculate benchmark percentile
        similar_strengths = [p["strength"] for p in self.LITIGATION_BENCHMARKS.values()]
        below_count = sum(1 for s in similar_strengths if s < strength)
        benchmark_percentile = (below_count / len(similar_strengths)) * 100

        return ClaimAnalysisResult(
            patent_number=patent_number,
            drug_name=drug_name,
            claim_breadth=claim_breadth,
            claim_breadth_score=breadth_score,
            independent_claims_count=independent_count,
            dependent_claims_count=dependent_count,
            overall_strength_score=float(strength),
            litigation_probability=75.0 if strength > 60 else 60.0,
            upheld_probability=float(upheld),
            vulnerabilities=vulnerabilities,
            vulnerability_severity=severity,
            prior_art_risk="medium",
            prior_art_concerns=[
                "Pre-filing publications in therapeutic area",
                "Related compound patents from competitors",
            ],
            similar_patents=similar,
            benchmark_percentile=benchmark_percentile,
            key_claims_summary=f"Patent covers {drug_name} with {independent_count} independent claims and {dependent_count} dependent claims. Claims appear to be {claim_breadth} in scope.",
            novelty_assessment="Mock analysis - novelty assessment requires full AI analysis with API key.",
            obviousness_assessment="Mock analysis - obviousness assessment requires full AI analysis with API key.",
            enablement_assessment="Mock analysis - enablement assessment requires full AI analysis with API key.",
            recommendations=[
                "Conduct detailed prior art search before trading decisions",
                "Monitor PTAB proceedings for inter partes review challenges",
                "Track related litigation in same therapeutic area",
            ],
            confidence_level="low",
        )

    def batch_analyze(
        self,
        patents: List[Dict[str, Any]],
    ) -> List[ClaimAnalysisResult]:
        """
        Analyze multiple patents.

        Args:
            patents: List of dicts with patent_number, drug_name, claims_text, metadata.

        Returns:
            List of ClaimAnalysisResult objects.
        """
        results = []

        for patent in patents:
            result = self.analyze_patent_claims(
                patent_number=patent["patent_number"],
                drug_name=patent["drug_name"],
                claims_text=patent.get("claims_text", ""),
                patent_metadata=patent.get("metadata"),
            )
            results.append(result)

        return results

    def compare_to_benchmarks(
        self,
        result: ClaimAnalysisResult,
    ) -> Dict[str, Any]:
        """
        Compare analysis result to known litigation benchmarks.

        Args:
            result: Analysis result to compare.

        Returns:
            Comparison data with insights.
        """
        comparisons = []

        for drug, benchmark in self.LITIGATION_BENCHMARKS.items():
            similarity = 100 - abs(result.overall_strength_score - benchmark["strength"])
            comparisons.append({
                "drug": drug.replace("_", " ").title(),
                "benchmark_strength": benchmark["strength"],
                "outcome": benchmark["outcome"],
                "similarity_score": similarity,
                "our_strength_delta": result.overall_strength_score - benchmark["strength"],
            })

        # Sort by similarity
        comparisons.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Most similar patents
        most_similar = comparisons[:3]

        # Prediction based on similar outcomes
        upheld_count = sum(
            1 for c in most_similar
            if c["outcome"] in ["UPHELD", "ONGOING"]
        )
        invalidated_count = sum(
            1 for c in most_similar
            if c["outcome"] in ["INVALIDATED", "SETTLED"]
        )

        predicted_outcome = "UPHELD" if upheld_count > invalidated_count else "INVALIDATED/SETTLED"

        return {
            "patent_number": result.patent_number,
            "drug_name": result.drug_name,
            "our_strength_score": result.overall_strength_score,
            "benchmark_percentile": result.benchmark_percentile,
            "most_similar_cases": most_similar,
            "predicted_outcome": predicted_outcome,
            "prediction_confidence": max(upheld_count, invalidated_count) / len(most_similar) * 100,
            "all_comparisons": comparisons,
        }


def generate_sample_analyses() -> List[ClaimAnalysisResult]:
    """
    Generate sample analyses for top drugs.

    Returns:
        List of sample analysis results.
    """
    analyzer = PatentClaimAnalyzer()

    # Sample patent data for major drugs
    sample_patents = [
        {
            "patent_number": "6090382",
            "drug_name": "Humira (adalimumab)",
            "claims_text": """
1. A human antibody, or an antigen-binding portion thereof, that binds to human TNF-alpha, comprising:
a) a light chain comprising a variable region having the amino acid sequence SEQ ID NO: 1; and
b) a heavy chain comprising a variable region having the amino acid sequence SEQ ID NO: 2.

2. The antibody of claim 1, wherein the antibody is a recombinant antibody.

3. A pharmaceutical composition comprising the antibody of claim 1 and a pharmaceutically acceptable carrier.

4. A method of treating rheumatoid arthritis comprising administering to a patient in need thereof a therapeutically effective amount of the antibody of claim 1.
            """,
            "metadata": {
                "filing_date": "1996-09-13",
                "issue_date": "2000-07-18",
                "expiration_date": "2023-01-31",
                "patent_type": "COMPOSITION",
                "therapeutic_area": "Immunology",
                "assignee": "Abbott Laboratories",
            }
        },
        {
            "patent_number": "7371746",
            "drug_name": "Eliquis (apixaban)",
            "claims_text": """
1. A compound having the formula:
[pyrazole-carboxamide structure with specific substituents]
or a pharmaceutically acceptable salt thereof.

2. The compound of claim 1, wherein said compound is 1-(4-methoxyphenyl)-7-oxo-6-[4-(2-oxopiperidin-1-yl)phenyl]-4,5,6,7-tetrahydro-1H-pyrazolo[3,4-c]pyridine-3-carboxamide.

3. A pharmaceutical composition comprising the compound of claim 1 and a pharmaceutically acceptable carrier.

4. The pharmaceutical composition of claim 3 for use in treating or preventing thromboembolic disorders.
            """,
            "metadata": {
                "filing_date": "2005-12-28",
                "issue_date": "2008-05-13",
                "expiration_date": "2026-12-31",
                "patent_type": "COMPOSITION",
                "therapeutic_area": "Cardiovascular",
                "assignee": "Bristol-Myers Squibb",
            }
        },
        {
            "patent_number": "8354509",
            "drug_name": "Keytruda (pembrolizumab)",
            "claims_text": """
1. An isolated anti-PD-1 antibody comprising:
(a) a heavy chain variable region comprising CDR1, CDR2, and CDR3 having amino acid sequences SEQ ID NOs: 1, 2, and 3, respectively; and
(b) a light chain variable region comprising CDR1, CDR2, and CDR3 having amino acid sequences SEQ ID NOs: 4, 5, and 6, respectively.

2. The antibody of claim 1, wherein the antibody is a humanized antibody.

3. A pharmaceutical composition comprising the antibody of claim 1 and a pharmaceutically acceptable carrier.

4. A method of treating cancer in a patient comprising administering to the patient a therapeutically effective amount of the antibody of claim 1.
            """,
            "metadata": {
                "filing_date": "2010-03-04",
                "issue_date": "2013-01-15",
                "expiration_date": "2028-07-28",
                "patent_type": "COMPOSITION",
                "therapeutic_area": "Oncology",
                "assignee": "Merck Sharp & Dohme",
            }
        },
        {
            "patent_number": "7863278",
            "drug_name": "Ibrance (palbociclib)",
            "claims_text": """
1. A compound of formula I:
[pyrido[2,3-d]pyrimidin-7-one structure]
wherein R1 is cyclopentyl or a pharmaceutically acceptable salt thereof.

2. 6-acetyl-8-cyclopentyl-5-methyl-2-{[5-(piperazin-1-yl)pyridin-2-yl]amino}pyrido[2,3-d]pyrimidin-7(8H)-one or a pharmaceutically acceptable salt thereof.

3. A pharmaceutical composition comprising the compound of claim 1 and a pharmaceutically acceptable carrier.

4. The compound of claim 2 for use in treating breast cancer.
            """,
            "metadata": {
                "filing_date": "2005-07-15",
                "issue_date": "2011-01-04",
                "expiration_date": "2023-11-17",
                "patent_type": "COMPOSITION",
                "therapeutic_area": "Oncology",
                "assignee": "Pfizer Inc.",
            }
        },
        {
            "patent_number": "6197819",
            "drug_name": "Lyrica (pregabalin)",
            "claims_text": """
1. A method of treating pain comprising administering to a patient in need thereof an effective amount of (S)-(+)-3-aminomethyl-5-methylhexanoic acid or a pharmaceutically acceptable salt thereof.

2. The method of claim 1, wherein the pain is neuropathic pain.

3. A pharmaceutical composition comprising (S)-(+)-3-aminomethyl-5-methylhexanoic acid and a pharmaceutically acceptable carrier.

4. The composition of claim 3 in a unit dosage form.
            """,
            "metadata": {
                "filing_date": "1997-07-17",
                "issue_date": "2001-03-06",
                "expiration_date": "2018-12-30",
                "patent_type": "METHOD_OF_USE",
                "therapeutic_area": "Neurology",
                "assignee": "Warner-Lambert Company",
            }
        },
    ]

    results = []
    for patent in sample_patents:
        result = analyzer.analyze_patent_claims(
            patent_number=patent["patent_number"],
            drug_name=patent["drug_name"],
            claims_text=patent["claims_text"],
            patent_metadata=patent["metadata"],
        )
        results.append(result)

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AI Patent Claim Strength Analysis - Sample Results")
    print("=" * 70)

    # Generate sample analyses
    results = generate_sample_analyses()

    for result in results:
        print(f"\n{'=' * 50}")
        print(f"Patent: {result.patent_number} - {result.drug_name}")
        print(f"{'=' * 50}")
        print(f"Overall Strength Score: {result.overall_strength_score:.1f}%")
        print(f"Claim Breadth: {result.claim_breadth} ({result.claim_breadth_score:.0f}/100)")
        print(f"Litigation Probability: {result.litigation_probability:.1f}%")
        print(f"Upheld Probability: {result.upheld_probability:.1f}%")
        print(f"Vulnerability Severity: {result.vulnerability_severity}")
        print(f"Prior Art Risk: {result.prior_art_risk}")
        print(f"Benchmark Percentile: {result.benchmark_percentile:.0f}%")

        print(f"\nVulnerabilities:")
        for vuln in result.vulnerabilities:
            print(f"  - {vuln}")

        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")

        print(f"\nKey Claims Summary:")
        print(f"  {result.key_claims_summary[:200]}...")

    # Show benchmark comparison for first result
    analyzer = PatentClaimAnalyzer()
    comparison = analyzer.compare_to_benchmarks(results[0])

    print("\n" + "=" * 70)
    print("Benchmark Comparison for", results[0].drug_name)
    print("=" * 70)
    print(f"Predicted Outcome: {comparison['predicted_outcome']}")
    print(f"Prediction Confidence: {comparison['prediction_confidence']:.0f}%")
    print("\nMost Similar Cases:")
    for case in comparison["most_similar_cases"]:
        print(f"  - {case['drug']}: {case['outcome']} (strength: {case['benchmark_strength']})")
