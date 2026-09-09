import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import {
  Zap, Brain, Database, Network, Clock, DollarSign, Hash,
  CheckCircle2, XCircle, History, BarChart2, Moon, Sun,
  Sparkles, TrendingDown, Award, ChevronDown, BookOpen,
  Layers, GitMerge,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || '';

/* ─── Pipeline registry (single source of truth) ─── */
const PIPELINE_KEYS    = ['llm_only', 'basic_rag', 'graphrag'];
const PIPELINE_COLORS  = { llm_only: '#ef4444', basic_rag: '#f97316', graphrag: '#16a34a' };
const PIPELINE_LABELS  = { llm_only: 'LLM-Only', basic_rag: 'Basic RAG', graphrag: 'GraphRAG' };
const PIPELINE_ICONS   = { llm_only: Brain, basic_rag: Database, graphrag: Network };
const PIPELINE_DESC    = {
  llm_only:  'No retrieval — pure parametric knowledge',
  basic_rag: 'FAISS vector search · top-5 chunks',
  graphrag:  'TigerGraph multi-hop · num_hops=2',
};

/* ─── Featured questions — each entity verified to exist in the LIVE TigerGraph graph
   and the answer confirmed on-topic. No pre-computed reduction numbers here: the only
   percentages shown anywhere are the genuine ones measured live after a query runs. ─── */
const FEATURED_QUESTIONS = [
  {
    label: 'Peremptory Challenges',
    icon: '⚖️',
    question: 'How did the court in People v. Kern (5987526) apply the principle from People v. Stiff (6000712) regarding the deference given to a trial court\'s determination that race-neutral explanations for peremptory challenges were pretextual?',
    answer: 'The court in People v. Kern affirmed the Supreme Court\'s determination that defense counsel\'s explanations for peremptory challenges were pretextual, stating that this determination is entitled to great deference on appeal and will not be disturbed when supported by the record, a principle consistent with the dissenting opinion in People v. Stiff, which argued that the trial court did not err in deeming a juror\'s relationships too remote to be anything but pretextual and that the court\'s conclusion was borne out by defense counsel\'s misleading suggestions.',
    hint: '2-hop . cases 5987526, 6000712',
  },
  {
    label: 'Codefendant Defense',
    icon: '🧑‍🤝‍🧑',
    question: 'What common legal issue did both People v. Wynn (5972547) and People v. Contes (5966890) address regarding a codefendant\'s defense, and what was the outcome in both cases?',
    answer: 'Both People v. Wynn and People v. Contes addressed the legal issue of whether the People satisfactorily disproved a codefendant\'s justification defense, and in both cases, the court rejected this claim, finding it to be without merit.',
    hint: '2-hop . cases 5972547, 5966890',
  },
  {
    label: 'Prosecutorial Summation',
    icon: '🗣️',
    question: 'How did the court in People v. Bailey (5988028) apply the principle from People v. Moore (6064559) regarding prosecutorial summation comments, specifically when defense counsel attacked witness credibility?',
    answer: 'In People v. Moore, the court affirmed that several allegedly improper prosecutorial statements on summation were fair responses to defense counsel\'s comments regarding the credibility of prosecution witnesses. Similarly, in People v. Bailey, the court applied this principle by finding that the prosecutor\'s review of an eyewitness\'s testimony and stress on its consistency with police accounts was a permissible response to defense counsel\'s repeated attacks on that witness\'s credibility, despite an isolated improper comment that the witness was "telling the truth."',
    hint: '2-hop . cases 6064559, 5988028',
  },
  {
    label: 'Verdict Sheet Objections',
    icon: '📋',
    question: 'How did the court in People v. Gladman (5905980) apply the principle regarding the preservation of objections to verdict sheets, as established or referenced in People v. Ervin (5920424), to the defendant\'s appeal?',
    answer: 'The court in People v. Gladman applied the principle by stating that the defendant\'s contention regarding the submission of verdict sheets was unpreserved for appellate review because he failed to object to their submission at trial, mirroring the reasoning in People v. Ervin which also found such a contention unpreserved due to the lack of an objection.',
    hint: '2-hop . cases 5905980, 5920424',
  },
  {
    label: 'Consent vs Probable Cause',
    icon: '🔍',
    question: 'What principle regarding consent as a substitute for probable cause originated in People v. Hodge (5956508), how did People v. Langdon (5989323) apply it, and how did People v. Robert C. W. (5992365) in turn build on or distinguish People v. Langdon\'s application of it?',
    answer: 'The principle that consent is a valid substitute for probable cause originated in People v. Hodge, where the court found no merit to the defendant\'s argument that his oral statements were the product of custodial detention without probable cause because he voluntarily agreed to accompany the officer. People v. Langdon applied this principle by affirming a youthful offender adjudication, finding that the interview of the defendant and subsequent transport to the scene of a burglary were consensual and proper. People v. Robert C. W. did not directly build on or distinguish People v. Langdon\'s application of the consent principle; instead, People v. Robert C. W. focused on the separate issue of when the right to counsel attaches, specifically whether the mere assignment of counsel constitutes actual representation precluding interrogation on an unrelated charge, a point also touched upon in People v. Langdon regarding the right to counsel, but distinct from the consent issue.',
    hint: '3-hop . cases 5956508, 5989323, 5992365',
  },
];

const DOMAIN_QUESTIONS = [
  {
    domain: '⚖️ Criminal Procedure (2-hop)',
    questions: [
      { question: 'How did the court in The Gas and Water Company of Downingtown v. The Borough of Downingtown (6245513) apply the principle regarding the sufficiency of an act\'s title, as discussed in Kittanning v. Mast (6274415), to determine the validity of the plaintiff\'s title to rights and franchises?', answer: 'In Kittanning v. Mast, the court discussed the principle that a legislative act\'s title does not need to be an index of its contents, but rather needs only to fairly give notice of the subject matter so as not to mislead. The Gas and Water Company of Downingtown v. The Borough of Downingtown applied this principle by upholding the validity of the plaintiff\'s title to rights and franchises, which was derived from a sheriff\'s sale under the Act of May 25, 1878. The court reasoned that despite the act\'s title not specifically mentioning gas and water companies, it was a supplement extending the provisions of an earlier act to various companies, and the act itself clearly included gas and water companies in its first section. Therefore, the title was deemed sufficient to give notice of the act\'s scope, consistent with the standard articulated in Kittanning v. Mast.' },
      { question: 'How did the court in Mulder v Donaldson, Lufkin & Jenrette (5997968) apply the principle of "law of the case" doctrine, as referenced in Krolick v DeGraff (6027842), to determine whether a punitive damages claim could be compelled to arbitration?', answer: 'The court in Mulder v Donaldson, Lufkin & Jenrette did not apply the "law of the case" doctrine to determine whether a punitive damages claim could be compelled to arbitration. Instead, it cited Mulder v Donaldson, Lufkin & Jenrette as an example of a case where the "law of the case" doctrine did not preclude consideration of a breach of fiduciary duty cause of action because the issue had not been resolved on the merits in a prior appeal. The primary issue in Mulder v Donaldson, Lufkin & Jenrette was whether the IAS Court properly denied defendant’s motion to compel arbitration of plaintiff’s punitive damages claim in light of the United States Supreme Court decision in Mastrobuono v Shearson Lehman Hutton, not the application of the "law of the case" doctrine to that specific arbitration issue.' },
      { question: 'How did the Pennsylvania Superior Court\'s application of the "clear and convincing evidence" standard, as established in Santosky v. Kramer (6305374), differ between the case involving L.A.J. and the case involving Scott and Tommy, regarding the necessity of new evidentiary hearings?', answer: 'In the case involving L.A.J., the Pennsylvania Superior Court reversed and remanded for the application of the "clear and convincing evidence" standard without explicitly addressing the necessity of new evidentiary hearings, as the decision in Santosky v. Kramer was issued while the case was pending on appeal. However, in the case involving Scott and Tommy, the court explicitly stated that new evidentiary hearings may not be necessary and left it to the trial court to determine, after hearing arguments of counsel, whether new evidentiary hearings were required, or if the court could simply reconsider its prior findings in light of the clear and convincing standard.' },
      { question: 'What was the ultimate procedural outcome for Harrison Graham\'s death sentence appeal, considering the Pennsylvania Supreme Court\'s decision in COMMONWEALTH of Pennsylvania, Appellee, v. Harrison GRAHAM, Appellant (1985671) and the subsequent order in Commonwealth v. Graham (6265216)?', answer: 'Despite Harrison Graham\'s desire to dismiss his appeal, the Pennsylvania Supreme Court in COMMONWEALTH of Pennsylvania, Appellee, v. Harrison GRAHAM, Appellant affirmed his death sentences due to the statutory requirement for automatic review in capital cases, and subsequently, the Pennsylvania Supreme Court in Commonwealth v. Graham granted a stay of execution pending the United States Supreme Court\'s decision on his petition for a writ of certiorari.' },
      { question: 'How did the court\'s reasoning for reversing the conviction in People v. Varona (5930500) relate to the basis for reversal in People v. Gray (5948198), given that both cases were decided by the same judge on the same date for the same crime?', answer: 'The court in People v. Varona reversed the conviction specifically due to the trial court\'s refusal to permit defendants to offer expert testimony regarding the complaining witness\'s mental capacity, citing the memorandum decision in the appeal of codefendant Jerome Dudley. People v. Gray, decided by the same judge on the same date for the same crime, was also reversed for the reasons set forth in the memorandum decision in the appeal of codefendant Jerome Dudley, indicating that the same issue regarding expert testimony on the complaining witness\'s mental capacity was the common ground for reversal in both cases.' },
      { question: 'What common legal issue did both People v. Wynn (5972547) and People v. Contes (5966890) address regarding a codefendant\'s defense, and what was the outcome in both cases?', answer: 'Both People v. Wynn and People v. Contes addressed the legal issue of whether the People satisfactorily disproved a codefendant\'s justification defense, and in both cases, the court rejected this claim, finding it to be without merit.' },
      { question: 'How did the court in People v. Lopez (6044208) distinguish its application of the preservation rule from People v. Lopez (6055357) regarding challenges to the factual sufficiency of a plea allocution, and what specific circumstance allowed for this distinction?', answer: 'While People v. Lopez (6055357) affirmed that a challenge to the factual sufficiency of a plea allocution is not preserved without a motion to withdraw the plea or vacate the judgment, People v. Lopez (6044208) recognized an exception to this rule. The court in People v. Lopez (6044208) found that a direct appeal challenging the sufficiency of the allocution was permissible, despite the absence of a formal post-allocution motion, because the defendant explicitly denied an essential element of the crime (that his ability to operate the motor vehicle was impaired by alcohol) during the allocution for aggravated unlicensed operation of a motor vehicle, thereby negating an element of the crime and requiring further inquiry from the court.' },
      { question: 'How did the court in People v. Sloan (5983908) implicitly distinguish its approach to evaluating the impact of a defendant\'s absence during voir dire from the dissenting judge\'s perspective in People v. Stiff (6000712) regarding the necessity of a detailed record for appellate review of juror challenges?', answer: 'In People v. Sloan, the court affirmed the judgment despite an off-the-record side-bar voir dire with one prospective juror, reasoning that the substance of the discussion was later disclosed in open court and the juror was peremptorily challenged, thus the defendant\'s absence did not substantially affect his defense. However, for a second off-the-record side-bar where no substance was disclosed, the court found the record insufficient for appellate review and declined to remit for a reconstruction hearing. This contrasts with the dissenting judge\'s detailed analysis in People v. Stiff, where the judge meticulously recounted defense counsel\'s misleading statements and lack of voir dire regarding a challenged juror\'s connections, using these specific record details to argue that the trial court did not err in deeming the challenge pretextual, thereby emphasizing the importance of a clear and complete record for evaluating the legitimacy of juror challenges on appeal.' },
    ],
  },
  {
    domain: '📜 Civil, Contract & Property Law (2-hop)',
    questions: [
      { question: 'How did the court in Rosenthal v. Elirlicher (6271651) apply the principle, recognized in Catlin v. Robinson (6234518), regarding the time limit for opening adversely obtained judgments, to justify granting a new trial nunc pro tunc more than two years after the original verdict?', answer: 'The court in Rosenthal v. Elirlicher acknowledged the established principle, recognized in Catlin v. Robinson, that the power to open adversely obtained judgments generally ceases with the term at which they are entered. However, it found an exception in this case due to the unique circumstances where the failure to properly prepare the record for review was attributed to erroneous ideas of practice, which were only corrected by later decisions. The Supreme Court had previously suggested that this was a proper case for relief by the court below, and the defendant had allowed the matter to rest upon an assurance of relief. Therefore, the court exercised its discretion to vacate the judgment and grant a new trial nunc pro tunc, despite the significant passage of time, to rectify a procedural error that prevented a proper review of the merits.' },
      { question: 'How did the court in People v. Frederick (5947537) apply the principle from People v. Frederick (5996423) regarding the sufficiency of the record to refute a claim of ineffective assistance of counsel, specifically concerning the defendant\'s satisfaction with their legal representation?', answer: 'In People v. Frederick (5947537), the court applied the principle from People v. Frederick (5996423) by affirming that a defendant\'s belated claim of ineffective assistance of counsel is refuted by the plea proceeding record where the defendant expressed satisfaction with counsel, similar to how the earlier People v. Frederick found the claim refuted by the record showing the defendant admitted satisfaction with their attorneys\' advice and representation.' },
      { question: 'How did the court in People v. Bailey (5988028) apply the principle from People v. Moore (6064559) regarding prosecutorial summation comments, specifically when defense counsel attacked witness credibility?', answer: 'In People v. Moore, the court affirmed that several allegedly improper prosecutorial statements on summation were fair responses to defense counsel\'s comments regarding the credibility of prosecution witnesses. Similarly, in People v. Bailey, the court applied this principle by finding that the prosecutor\'s review of an eyewitness\'s testimony and stress on its consistency with police accounts was a permissible response to defense counsel\'s repeated attacks on that witness\'s credibility, despite an isolated improper comment that the witness was "telling the truth."' },
      { question: 'How did the court in People v. Kern (5987526) apply the principle from People v. Stiff (6000712) regarding the deference given to a trial court\'s determination that race-neutral explanations for peremptory challenges were pretextual?', answer: 'The court in People v. Kern affirmed the Supreme Court\'s determination that defense counsel\'s explanations for peremptory challenges were pretextual, stating that this determination is entitled to great deference on appeal and will not be disturbed when supported by the record, a principle consistent with the dissenting opinion in People v. Stiff, which argued that the trial court did not err in deeming a juror\'s relationships too remote to be anything but pretextual and that the court\'s conclusion was borne out by defense counsel\'s misleading suggestions.' },
      { question: 'How did the court in People v. Freeman (5966590) expand upon the reasoning from People v. Contes (6005869) regarding the improper jury instruction about "reasonable degree of certainty"?', answer: 'While People v. Contes found the jury instruction that "it is possible to establish the guilt of a defendant charged with a crime to a reasonable degree of certainty. To that degree of proof, the People must be held" to be an improper reduction of the People\'s burden of proof, People v. Freeman further specified that this error in the charge deprived the defendant of their Fifth Amendment right to a verdict of guilt beyond a reasonable doubt, citing additional Supreme Court precedent like Sullivan v Louisiana and Cage v Louisiana to support this constitutional implication.' },
      { question: 'How did the court in Delehanty, S. (6156400) apply the principle from Frankenthaler, S. (6161070) regarding the apportionment of estate taxes for annuities, and what additional guidance did Delehanty, S. provide concerning the reimbursement method?', answer: 'The court in Delehanty, S. applied the principle from Frankenthaler, S. by stating that the tax on the value of annuities must be paid initially from the fund set aside to produce them, and this payment is to be reimbursed to the estate according to the rule in Matter of Tracy. Delehanty, S. further clarified that if the parties voluntarily agree upon a method for reimbursement that eases the burden on the annuitants, the court will approve such a consented program.' },
    ],
  },
  {
    domain: '🏛️ Constitutional & Family Law (2-hop)',
    questions: [
      { question: 'How did the court in Jones v. Wagner (6238956) interpret the phrase "shall do as little damage to the surface as possible" in a deed reserving mineral rights, and how does this interpretation relate to the three classes of cases concerning surface support waivers identified in Jones v. Wagner (6251038)?', answer: 'In Jones v. Wagner, the court interpreted the phrase "shall do as little damage to the surface as possible" as applying to the surface rights indispensable for mining operations, such as making explorations, boring holes, sinking shafts, and erecting structures, rather than implying a waiver of the absolute right to surface support. This interpretation places the case within the first class of cases identified in the earlier Jones v. Wagner opinion, where there is no express or implied waiver of damages to the surface, thus entitling the surface owner to recover compensation for injuries sustained due to a failure to properly support the surface.' },
      { question: 'How did the court in Mercado v. City of New York (6067592) apply the principle regarding a partial directed verdict, as established in Szczerbiak v Pilat (2022154), to the specific facts of the medical malpractice/wrongful death action before it?', answer: 'In Mercado v. City of New York, the court applied the principle from Szczerbiak v Pilat, which concerns the standard for granting a directed verdict, by granting the defendant\'s motion for a partial directed verdict. This was done because, even affording the plaintiffs every favorable inference, there was no rational process by which the triers of fact could have found that the defendant had prescribed Macrodantin for the decedent, thus precluding all reference to the drug.' },
      { question: 'How did the court in Matter Torres v. Coughlin (6019606) apply the principle of substantial evidence for possession of a weapon, as established in Matter Bryant v. Coughlin (6047146), to a situation where other inmates had access to the area where the weapon was found?', answer: 'In Matter Torres v. Coughlin, the court applied the principle of substantial evidence for possession of a weapon by determining that the discovery of a metal shank under the petitioner\'s locker, an area over which he had control, was sufficient to create a reasonable inference of possession, even though other inmates had access to that area. This aligns with the general principle of substantial evidence for weapon possession affirmed in Matter Bryant v. Coughlin, which found that misbehavior reports and correction officer testimony constituted substantial evidence for a weapon possession violation.' },
      { question: 'How did the court in Co. v. City of New York (5932666) implicitly distinguish the type of fraud claims that were dismissed in Murtha v. Yonkers Child Care Assn. (6045687) when it affirmed the denial of leave to amend the complaint?', answer: 'In Co. v. City of New York, the court affirmed the denial of leave to amend the complaint by stating that the proposed fraud claims were legally deficient because they relied upon alleged misrepresentations of future intent and failed to plead fraud with sufficient particularity, and also because a cause of action for fraud does not arise when the only fraud alleged relates to a breach of contract. This implicitly distinguishes the fraud claims in Murtha v. Yonkers Child Care Assn., where the court simply stated that the plaintiff\'s allegations, even if true, failed to set forth the requisite elements to support viable claims for fraud or tortious interference with contract, without specifying the particular deficiencies related to future intent, particularity, or the relationship to a breach of contract.' },
      { question: 'How did the court in Mansour v. Abrams (5909013) apply the principle of requiring discovery for proof of malice, as established in Mansour v. Abrams (5979990), to the specific context of a tortious interference claim against defendant Marcus?', answer: 'The court in Mansour v. Abrams (5909013) applied the principle from Mansour v. Abrams (5979990) by stating that plaintiff was entitled to previously ordered discovery to explore Marcus\'s motivation because plaintiff must prove that Marcus acted in bad faith to prove his cause of action for tortious interference with an at-will contract, thereby linking the need for discovery to establish malice (or bad faith in this context) to a specific defendant and cause of action.' },
      { question: 'How did the court in People v. Gladman (5905980) apply the principle regarding the preservation of objections to verdict sheets, as established or referenced in People v. Ervin (5920424), to the defendant\'s appeal?', answer: 'The court in People v. Gladman applied the principle by stating that the defendant\'s contention regarding the submission of verdict sheets was unpreserved for appellate review because he failed to object to their submission at trial, mirroring the reasoning in People v. Ervin which also found such a contention unpreserved due to the lack of an objection.' },
      { question: 'How did the court in Hall v. Superior Court (6106766) distinguish the application of attorney fee awards in the context of a prevailing party, as compared to the circumstances that led to the attorney fee award in LAWRENCE PASTERNACK v. THOMAS B. MCCULLOUGH, JR. (4698850)?', answer: 'In LAWRENCE PASTERNACK v. THOMAS B. MCCULLOUGH, JR., the attorney fee award was based on a statutory entitlement for a prevailing defendant on a special motion to strike under the anti-SLAPP statute, where the court determined the reasonable market value of the attorneys\' services. In contrast, Hall v. Superior Court distinguished its situation by finding that Hall was not a successful party entitled to attorney fees because the relief obtained in Hall I was already granted by the trial court and available from the Department of Motor Vehicles, thus not providing any new or additional relief that would qualify him as a prevailing party for a fee award.' },
      { question: 'How did the court in People v. Whalen (5905090) apply the principle of harmless error regarding the operability instruction, as established in cases like People v. Henry (6024019), to the specific facts of its case?', answer: 'In People v. Whalen, the court applied the harmless error principle by finding that despite the trial court\'s failure to instruct the jury on limiting deliberations to the operability of the weapon at the time of the incident, the error was harmless because the evidence of the defendant\'s possession of the weapon and its operability was overwhelming. This mirrors the reasoning in People v. Henry, where the court also found the failure to deliver an operability instruction to be harmless error due to overwhelming evidence of operability and the absence of any contested issue concerning that element.' },
      { question: 'How did the court in the later case, People v. Contes (5949510), apply the legal sufficiency standard from the earlier case, People v. Contes (6011888), to determine whether the complainant suffered "physical injury" in the context of a robbery?', answer: 'The court in People v. Contes (5949510) applied the legal sufficiency standard, which requires viewing the evidence in the light most favorable to the prosecution, to find that the complainant suffered "physical injury" under Penal Law § 10.00 (9). This was established by evidence that the defendant knocked the complainant down, grabbed her neck, caused her to fall onto subway steps, resulting in a bruised elbow requiring paramedic treatment and a lasting scar, and "real pain" in her lower back for approximately one month, with the scar alone and the duration of pain being sufficient to constitute physical injury.' },
      { question: 'How did the court in *State v. Starr* (4656754) clarify the definition of "costs" in RCW 10.01.160(2) in a way that would have been relevant to the sentencing court\'s decision regarding discretionary legal financial obligations for Alexander James Huckins in *State v. Huckins* (4315129), particularly concerning community custody supervision fees?', answer: 'In *State v. Starr*, the court clarified that community custody supervision fees are not "costs" as defined by RCW 10.01.160(2). This clarification would be relevant to *State v. Huckins* because it establishes that such fees are not subject to the same indigency waivers as other discretionary legal financial obligations, meaning that even if Huckins were found indigent, community custody supervision fees would not automatically be waived as "costs."' },
      { question: 'How did the court in Vasile v. Vasile (5964594) apply the principle regarding the termination of visitation rights, as referenced in Matter Reed v. Crim (5991610), to the specific facts of its case?', answer: 'The court in Vasile v. Vasile applied the principle that termination of visitation requires compelling reasons and substantial evidence of detriment to the child\'s welfare by finding that the respondent\'s use of excessive physical force on the children, repeated disregard of court orders, and use of the children as pawns in disputes with his ex-wife constituted substantial evidence justifying the termination of visitation rights.' },
      { question: 'How did the court in Smith v. Hay (6272238) apply the principle from Mullet v. Hensel (6272488) regarding the sound discretion of the court in opening a judgment, and what specific factor in Smith v. Hay led to a partial modification of the judgment, unlike the complete refusal to open in Mullet v. Hensel?', answer: 'The court in Smith v. Hay applied the principle from Mullet v. Hensel that an application to open a judgment is addressed to the sound discretion of the court and that the appellate question is whether that discretion was properly exercised. While Smith v. Hay, like Mullet v. Hensel, generally refused to open the judgment based on insufficient evidence for the main contention, Smith v. Hay partially modified the judgment by directing a credit of $18.00 because the plaintiff conceded in his sworn answer that this amount, representing one year\'s interest, had been paid and was erroneously included in the judgment. In contrast, Mullet v. Hensel found no evidence worthy of consideration for fraud and indefinite testimony regarding payment, leading to a complete refusal to open the judgment without any modification.' },
      { question: 'What specific type of neglect was present in both Linker-Flores v. Ark. Dep\'t of Human Servs. (6110142) and Linker-Flores v. Ark. Dep\'t of Human Servs. (6109471) that contributed to the children being adjudicated dependent-neglected?', answer: 'In both Linker-Flores v. Ark. Dep\'t of Human Servs. (6110142) and Linker-Flores v. Ark. Dep\'t of Human Servs. (6109471), environmental neglect was a contributing factor to the children being adjudicated dependent-neglected.' },
    ],
  },
  {
    domain: '🔗 Multi-Case Citation Chains (3-hop)',
    questions: [
      { question: 'What principle regarding common-law indemnification for property owners without control over a worksite originated in Kosiorek v. Bethlehem Steel Corp. (5922378), how did Bland v. Manocherian (5950920) apply this principle to grant indemnification to a construction manager and a subcontractor, and how did Bland v. Manocherian (5984358) then distinguish or build upon the earlier Bland v. Manocherian\'s application of indemnification by focusing solely on Labor Law § 240 (1) liability without addressing indemnification?', answer: 'Kosiorek v. Bethlehem Steel Corp. established the principle that a property owner whose liability is based solely on its status as owner and who had no control or supervision of the worksite is entitled to contractual indemnification. Bland v. Manocherian (1993) applied this principle by granting common-law indemnification to Balling Construction Management, Inc. and CRSS Constructors, Inc. (a joint venture) because they provided only contract management services and had no authority over contractors or the work, and to A.L.P. Steel Corp. because it subcontracted the erection work and did not control or supervise the work or direct construction procedures or safety measures. Bland v. Manocherian (1994) did not address indemnification, instead focusing solely on affirming partial summary judgment for the plaintiff on a Labor Law § 240 (1) cause of action, emphasizing the owner\'s non-delegable duty to provide proper protection regardless of the worker\'s actions.' },
      { question: 'What principle regarding the extent of cross-examination on immaterial matters originated in People v. Sorge (5953698), how did People v. Contes (5978611) apply this principle to limit cross-examination concerning a witness\'s alleged prior bank robbery, and how did People v. Boulware (6050541) in turn build on or distinguish People v. Contes\'s application by restricting cross-examination regarding a victim\'s arrests versus convictions?', answer: 'The principle that the extent of cross-examination of a witness upon matters immaterial to the issue is within the discretion of the trial court, and reviewable only for plain abuse and injustice, originated in People v. Sorge. People v. Contes applied this principle by affirming the trial court\'s discretion in limiting cross-examination of a prosecution witness concerning a bank robbery that the witness denied committing, and further in refusing to permit the introduction of an FBI videotape of that robbery to impeach the witness\'s credibility. People v. Boulware built on this by affirming the trial court\'s proper exercise of discretion in precluding questions concerning a victim\'s arrests as opposed to convictions, aligning with the general principle of limiting cross-examination to relevant matters and convictions for impeachment, rather than mere arrests.' },
      { question: 'What specific issue did People v. Lang (5935319) address regarding a defendant\'s intent, how did People v. Cruickshank (5953899) implicitly apply a similar standard to a different issue, and how did People v. Bleakley (6047154) then explicitly apply that standard to yet another distinct issue?', answer: 'People v. Lang addressed whether a defendant\'s intoxication prevented the formation of requisite intent, finding it to be an issue of fact and credibility for the jury. People v. Cruickshank, while not directly on intent, implicitly applied a similar standard by affirming the denial of youthful offender status after reviewing the record and considering relevant factors, indicating a factual determination. People v. Bleakley then explicitly applied this standard to the denial of youthful offender status, stating that the court did not abuse its discretion in light of the defendant\'s actions, thereby treating the youthful offender determination as a factual and discretionary matter for the court, similar to how Lang treated intent as a factual matter for the jury.' },
      { question: 'What was the original burden-shifting framework established in McDONNELL DOUGLAS CORP v. GREEN (108786) for proving employment discrimination, how did EVANS v. TECHNOLOGIES APPLICATIONS & SERVICE COMPANY (715797) apply this framework to a failure-to-promote claim, and how did DEPAOLI v. VACATION SALES ASSOCIATES, LLC (2481677) subsequently apply the framework to a retaliation claim, specifically regarding the plaintiff\'s burden to show a causal connection?', answer: 'In McDONNELL DOUGLAS CORP v. GREEN, the Supreme Court established a three-part burden-shifting framework for proving employment discrimination under Title VII: first, the plaintiff must establish a prima facie case of discrimination; second, the burden shifts to the employer to articulate a legitimate, nondiscriminatory reason for its action; and third, the plaintiff must then be afforded an opportunity to prove that the employer\'s stated reason was in fact pretext. EVANS v. TECHNOLOGIES APPLICATIONS & SERVICE COMPANY applied this framework to a failure-to-promote claim, finding that the plaintiff failed to establish a prima facie case because she did not show that she was qualified for the promotion or that the position remained open or was filled by someone outside the protected class. DEPAOLI v. VACATION SALES ASSOCIATES, LLC, in turn, applied the McDonnell Douglas framework to a retaliation claim, noting that to establish a prima facie case of retaliation, the plaintiff must show a causal connection between the protected activity and the adverse employment action, which can be demonstrated by evidence that the employer was aware of the protected activity and that the adverse action followed shortly thereafter.' },
      { question: 'What principle regarding consent as a substitute for probable cause originated in People v. Hodge (5956508), how did People v. Langdon (5989323) apply it, and how did People v. Robert C. W. (5992365) in turn build on or distinguish People v. Langdon\'s application of it?', answer: 'The principle that consent is a valid substitute for probable cause originated in People v. Hodge, where the court found no merit to the defendant\'s argument that his oral statements were the product of custodial detention without probable cause because he voluntarily agreed to accompany the officer. People v. Langdon applied this principle by affirming a youthful offender adjudication, finding that the interview of the defendant and subsequent transport to the scene of a burglary were consensual and proper. People v. Robert C. W. did not directly build on or distinguish People v. Langdon\'s application of the consent principle; instead, People v. Robert C. W. focused on the separate issue of when the right to counsel attaches, specifically whether the mere assignment of counsel constitutes actual representation precluding interrogation on an unrelated charge, a point also touched upon in People v. Langdon regarding the right to counsel, but distinct from the consent issue.' },
      { question: 'What principle regarding the dedication of streets to public use originated in Trutt v. Spotts (6238271), how did the Opinion (6238701) by Mr. Justice Green apply or affirm a related principle concerning already located streets, and how did the Opinion (6238833) by Mr. Justice Green then distinguish the concept of dedication when a grantor merely refers to a municipally laid out but unopened street as a boundary?', answer: 'Trutt v. Spotts established the principle that when a proprietor sells and conveys lots according to a plan showing them to be on streets, he is held to have stamped upon them the character of public streets, thereby dedicating them to public use and preventing him from revoking that dedication. The Opinion by Mr. Justice Green in 1886 affirmed the decision in In re Jackson street, holding that the act of 1874 does not apply to cases where streets in Philadelphia have already been located, thereby reinforcing the finality of established street locations. The Opinion by Mr. Justice Green in 1887 distinguished the concept of dedication by clarifying that merely referring to a street laid out but not opened by municipal authority as a boundary in a deed does not constitute a dedication of the land within the street limits to the public, nor does it deprive the owner of the right to compensation when the land is actually taken, as such a reference is a private contractual matter between grantor and grantee, not an act of the owner towards the public.' },
      { question: 'What principle regarding the admissibility of showup identification procedures originated in People v. Brnja (5904974), how did People v. Rivera (5912255) apply it in the context of a showup involving codefendants, and how did People v. Love (5990920) in turn build on or distinguish People v. Rivera\'s application of it?', answer: 'The principle originating in People v. Brnja is that showup identification testimony is admissible if the procedure occurred within a reasonably short time after the crime and immediately subsequent to the apprehension of a defendant fitting the description, and was not so unnecessarily suggestive and conducive to irreparable mistaken identification as to deny due process. People v. Rivera applied this by affirming the admissibility of a showup identification where the defendant and codefendant, both matching descriptions, were promptly detained and viewed on-scene by the complainant, specifically finding that viewing codefendants together was not unduly suggestive. People v. Love, however, does not explicitly build on or distinguish People v. Rivera\'s application regarding codefendants; instead, it generally affirms the denial of a motion to suppress identification testimony, citing other cases for the general proposition of proper pretrial identification, without detailing the specific nature of the showup or its suggestiveness.' },
      { question: 'What specific procedural requirement for persistent felony offender sentencing, first highlighted in People v. Kelly (5925781), was further elaborated upon in Mincey v. Arizona (5951652), and how did People v. Favor (5976484) demonstrate the People\'s burden in meeting that requirement?', answer: 'People v. Kelly established that a court must consider a defendant\'s "history and character" in addition to criminal conduct when determining persistent felony offender status, as required by CPL 400.20 (1) (b). Mincey v. Arizona further elaborated on this by stating that the court must provide a separate statement setting forth the dates and places of prior convictions and the factors in the defendant\'s history and background warranting persistent felony offender status, as required by CPL 400.20 (3), and must expressly adopt the conviction history or elaborate on the factors considered. People v. Favor demonstrated that the People bear the burden of proving that the defendant is the person convicted of the prior felonies set forth in the court\'s statement, and that a prior court decision alone, without the underlying proof being received as evidence, is insufficient to meet this burden.' },
      { question: 'What specific pleading requirement for municipal liability under 42 USC § 1983 was established in People v. Jackson (5960551), how did Felder v. Casey (5990590) apply this requirement to dismiss a claim, and how did the later Felder v. Casey (6070817) opinion, in turn, address the viability of a civil rights claim against a municipality in light of this requirement?', answer: 'In People v. Jackson, the court established that to prevail on a 42 USC § 1983 claim against a municipality, a plaintiff must specifically plead and prove that the municipality itself caused the constitutional violation, not merely through respondeat superior. Felder v. Casey (5990590) applied this by dismissing the plaintiff\'s claim against the municipality because of his failure to specifically plead the existence of an official policy or custom that deprived him of a constitutional right. The later Felder v. Casey (6070817) opinion, however, affirmed the denial of the defendants\' motion for summary judgment, allowing the civil rights claim to proceed, indicating that the plaintiff in that specific case had overcome the pleading hurdle regarding municipal liability, though the opinion itself does not detail how.' },
      { question: 'What principle regarding an executor\'s duty to make an estate productive originated in Wingate, S. (6154762), how did Wingate, S. (6154800) apply this principle to justify denying commissions, and how did Maximilian Moss, S. (6178451) subsequently address the executors\' discretion in liquidating a business, considering the potential for unproductivity?', answer: 'In Wingate, S., the court established that an executor cannot keep an estate substantially unproductive for an extended period, thereby depriving a dependent of support, and escape liability, emphasizing that a power of sale, though using the word "authorize," can be mandatory if the testator\'s intention was for prompt sale and distribution. Wingate, S. (144 Misc. 434) applied this principle by denying commissions to an executor who, despite a testamentary direction to sell and divide property, kept estate funds in an unproductive commercial bank account for seventeen years, failing to set up a trust or pay income to the beneficiary. Maximilian Moss, S. later addressed the executors\' discretion in liquidating a business, noting that the will directed liquidation "as soon as may be practicable" and granted wide discretionary powers, including the right to retain and invest in nonlegal investments, thereby distinguishing the level of discretion and the nature of the assets from the more straightforward sale and distribution directed in the Wingate cases.' },
      { question: 'What standard for viewing evidence in a light most favorable to the prosecution originated in People v. Contes (5925303), how did People v. Contes (6005367) apply this standard to determine the legal sufficiency of evidence for robbery in the third degree, and how did People v. Contes (6009013) subsequently apply the same standard to establish legal sufficiency for felony murder based on an underlying robbery?', answer: 'The standard for viewing evidence in a light most favorable to the prosecution originated in People v. Contes (60 NY2d 620), which People v. Contes (5925303) cited to affirm a conviction for robbery in the second degree. People v. Contes (6005367) applied this standard to find legally sufficient evidence for robbery in the third degree, specifically noting that the defendant\'s statement, "Do as I tell you and you won’t get hurt," was sufficient to conclude a threat of immediate physical force. People v. Contes (6009013) subsequently applied the same standard to determine that the evidence was legally sufficient to establish the defendant\'s guilt of felony murder, based upon the underlying felony of robbery.' },
    ],
  },
];

/* ─── Themes ─── */
const THEMES = {
  light: {
    pageBg: '#f0f4f8',
    heroBg: 'linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)',
    surface: '#ffffff', surface2: '#f8fafc', surfaceHover: '#f1f5f9',
    border: '#e2e8f0', borderStrong: '#cbd5e1',
    text: '#0f172a', textMuted: '#475569', textSubtle: '#94a3b8',
    inputBg: '#ffffff', metricBg: '#f8fafc',
    graphragBorder: '#16a34a',
    graphragBg: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
    graphragGlow: '0 0 0 2px #16a34a40, 0 8px 32px rgba(22,163,74,0.15)',
    errorBg: '#fef2f2', errorBorder: '#fca5a5', errorText: '#dc2626',
    badgePassBg: '#dcfce7', badgePassText: '#15803d',
    badgeFailBg: '#fee2e2', badgeFailText: '#dc2626',
    chartGrid: '#e2e8f0', tooltipBg: '#ffffff',
    spinnerTrack: '#e2e8f0', spinnerHead: '#16a34a',
    btnGradient: 'linear-gradient(135deg, #16a34a, #15803d)',
    btnDisabledBg: '#e2e8f0', btnDisabledText: '#94a3b8',
    tableRowBorder: '#f1f5f9',
    toggleBg: 'rgba(255,255,255,0.15)',
    tagBg: '#f1f5f9', tagText: '#475569',
    bertBg: 'linear-gradient(135deg, #eff6ff, #dbeafe)', bertBorder: '#93c5fd', bertText: '#1e40af',
    accentGlow: 'rgba(22,163,74,0.1)',
    cardShadow: '0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04)',
    reductionStat: '#15803d',
    statCard: 'rgba(255,255,255,0.12)', statCardBorder: 'rgba(255,255,255,0.2)',
    statText: '#ffffff', statSubtext: 'rgba(255,255,255,0.7)',
    sectionBg: '#f8fafc',
  },
  dark: {
    pageBg: '#0a0f1a',
    heroBg: 'linear-gradient(135deg, #0a0f1a 0%, #0d1f2d 50%, #0a1628 100%)',
    surface: '#111827', surface2: '#0f172a', surfaceHover: '#1e293b',
    border: '#1e293b', borderStrong: '#334155',
    text: '#f1f5f9', textMuted: '#94a3b8', textSubtle: '#475569',
    inputBg: '#111827', metricBg: '#0f172a',
    graphragBorder: '#22c55e',
    graphragBg: 'linear-gradient(135deg, #052e16 0%, #14532d 100%)',
    graphragGlow: '0 0 0 2px #22c55e40, 0 8px 32px rgba(34,197,94,0.2)',
    errorBg: '#450a0a', errorBorder: '#ef4444', errorText: '#fca5a5',
    badgePassBg: '#14532d', badgePassText: '#4ade80',
    badgeFailBg: '#450a0a', badgeFailText: '#f87171',
    chartGrid: '#1e293b', tooltipBg: '#111827',
    spinnerTrack: '#1e293b', spinnerHead: '#22c55e',
    btnGradient: 'linear-gradient(135deg, #22c55e, #16a34a)',
    btnDisabledBg: '#1e293b', btnDisabledText: '#475569',
    tableRowBorder: '#111827',
    toggleBg: 'rgba(255,255,255,0.1)',
    tagBg: '#1e293b', tagText: '#94a3b8',
    bertBg: 'linear-gradient(135deg, #0c1929, #0f2040)', bertBorder: '#3b82f6', bertText: '#93c5fd',
    accentGlow: 'rgba(34,197,94,0.08)',
    cardShadow: '0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.3)',
    reductionStat: '#4ade80',
    statCard: 'rgba(255,255,255,0.07)', statCardBorder: 'rgba(255,255,255,0.12)',
    statText: '#ffffff', statSubtext: 'rgba(255,255,255,0.6)',
    sectionBg: '#0f172a',
  },
};

/* ─── Animated counter ─── */
function AnimatedNumber({ value, duration = 800, suffix = '' }) {
  const [display, setDisplay] = useState(0);
  const prev = useRef(0);
  useEffect(() => {
    const start     = prev.current;
    const end       = parseFloat(value);
    const startTime = performance.now();
    function tick(now) {
      const p     = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(start + (end - start) * eased);
      if (p < 1) requestAnimationFrame(tick);
      else { prev.current = end; setDisplay(end); }
    }
    requestAnimationFrame(tick);
  }, [value, duration]);
  return (
    <>
      {typeof value === 'number' && !Number.isInteger(value)
        ? display.toFixed(1)
        : Math.round(display).toLocaleString()}
      {suffix}
    </>
  );
}

/* ─── JudgeBadge ─── */
function JudgeBadge({ judge, source, t }) {
  if (!judge) return null;
  const pass = judge === 'PASS';
  const sourceLabel = source === 'gemini_fallback' ? 'Gemini fallback'
    : source === 'huggingface' ? 'HuggingFace'
    : null;
  return (
    <span
      title={sourceLabel ? `Judged by: ${sourceLabel}` : undefined}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        background: pass ? t.badgePassBg : t.badgeFailBg,
        color: pass ? t.badgePassText : t.badgeFailText,
        borderRadius: 20, padding: '3px 10px', fontSize: 11, fontWeight: 700,
      }}
    >
      {pass ? <CheckCircle2 size={11} /> : <XCircle size={11} />}
      {judge}
      {sourceLabel && (
        <span style={{ opacity: 0.65, fontWeight: 600, fontSize: 9 }}>· {sourceLabel}</span>
      )}
    </span>
  );
}

/* ─── Token Bar ─── */
function TokenBar({ label, tokens, maxTokens, color, t }) {
  const pct = Math.min((tokens / maxTokens) * 100, 100);
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: t.textMuted }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 800, color }}>{tokens.toLocaleString()} tokens</span>
      </div>
      <div style={{ height: 8, borderRadius: 99, background: t.border, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color,
          borderRadius: 99, transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)',
        }} />
      </div>
    </div>
  );
}

/* ─── Spinner ─── */
function Spinner({ t }) {
  const [step, setStep] = useState(0);
  const steps = [
    { label: 'Querying LLM-Only…',   icon: Brain,    color: '#ef4444' },
    { label: 'Running Basic RAG…',   icon: Database, color: '#f97316' },
    { label: 'Traversing TigerGraph…', icon: Network, color: '#16a34a' },
  ];
  useEffect(() => {
    const id = setInterval(() => setStep(s => (s + 1) % steps.length), 2000);
    return () => clearInterval(id);
  }, []);
  const StepIcon = steps[step].icon;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '52px 24px', gap: 24 }}>
      <div style={{ position: 'relative', width: 72, height: 72 }}>
        <svg width="72" height="72" style={{ position: 'absolute', top: 0, left: 0, animation: 'spin 1.2s linear infinite' }}>
          <circle cx="36" cy="36" r="30" fill="none" stroke={t.spinnerTrack} strokeWidth="3" />
          <circle cx="36" cy="36" r="30" fill="none" stroke={steps[step].color} strokeWidth="3"
            strokeDasharray="50 140" strokeLinecap="round" />
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <StepIcon size={22} color={steps[step].color} />
        </div>
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: t.text, marginBottom: 6 }}>
          Running all 3 pipelines in parallel
        </div>
        <div style={{ fontSize: 13, color: t.textMuted, marginBottom: 16 }}>{steps[step].label}</div>
        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
          {PIPELINE_KEYS.map((key, i) => {
            const StepIcon = steps[i].icon;
            return (
              <div key={key} style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '4px 12px', borderRadius: 20,
                background: i === step ? `${steps[i].color}20` : t.surface2,
                border: `1px solid ${i === step ? steps[i].color : t.border}`,
                transition: 'all 0.3s',
              }}>
                <StepIcon size={11} color={i === step ? steps[i].color : t.textSubtle} />
                <span style={{ fontSize: 11, color: i === step ? steps[i].color : t.textSubtle, fontWeight: 600 }}>
                  {PIPELINE_LABELS[key]}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ─── Pipeline Card ─── */
function PipelineCard({ name, data, judge, judgeSource, t }) {
  const [expanded, setExpanded] = useState(true);
  const color  = PIPELINE_COLORS[name];
  const Icon   = PIPELINE_ICONS[name];
  const isGrag = name === 'graphrag';

  return (
    <div
      style={{
        background: isGrag ? t.graphragBg : t.surface,
        border: `1.5px solid ${isGrag ? t.graphragBorder : t.border}`,
        boxShadow: isGrag ? t.graphragGlow : t.cardShadow,
        borderRadius: 16, overflow: 'hidden',
        transition: 'transform 0.2s, box-shadow 0.2s',
      }}
      onMouseEnter={e => { if (!isGrag) e.currentTarget.style.transform = 'translateY(-2px)'; }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'none'; }}
    >
      <div style={{ height: 4, background: `linear-gradient(90deg, ${color}, ${color}99)` }} />

      <div style={{ padding: '16px 20px' }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              background: `${color}18`, borderRadius: 10,
              width: 36, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: `1px solid ${color}30`,
            }}>
              <Icon size={17} color={color} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontWeight: 800, fontSize: 15, color }}>{PIPELINE_LABELS[name]}</span>
                {isGrag && data.graph_context_found !== false && (
                  <span style={{
                    background: 'linear-gradient(135deg, #16a34a, #15803d)',
                    color: '#fff', borderRadius: 6, padding: '1px 7px',
                    fontSize: 10, fontWeight: 700, letterSpacing: '0.04em',
                  }}>BEST</span>
                )}
                {isGrag && data.graph_context_found === false && (
                  <span style={{
                    background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                    color: '#fff', borderRadius: 6, padding: '1px 7px',
                    fontSize: 10, fontWeight: 700, letterSpacing: '0.04em',
                  }}>NO MATCH</span>
                )}
              </div>
              <div style={{ fontSize: 11, color: t.textSubtle, marginTop: 1 }}>{PIPELINE_DESC[name]}</div>
            </div>
          </div>
          <JudgeBadge judge={judge} source={judgeSource} t={t} />
        </div>

        {/* Metrics */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 14 }}>
          {[
            { icon: Hash,        label: 'Tokens',  value: data.total_tokens.toLocaleString(), color },
            { icon: Clock,       label: 'Latency', value: `${data.latency_s}s`,              color: t.textMuted },
            { icon: DollarSign,  label: 'Cost',    value: `$${data.cost_usd.toFixed(5)}`,    color: t.textMuted },
          ].map(({ icon: MIcon, label, value, color: c }) => (
            <div key={label} style={{
              background: t.metricBg, border: `1px solid ${t.border}`,
              borderRadius: 10, padding: '10px 12px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3 }}>
                <MIcon size={10} color={t.textSubtle} />
                <span style={{ fontSize: 10, color: t.textSubtle, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
                  {label}
                </span>
              </div>
              <div style={{ fontSize: 15, fontWeight: 800, color: c }}>{value}</div>
            </div>
          ))}
        </div>

        {/* GraphRAG tags — reflects the actual citation-graph traversal, not a retriever
            type this pipeline doesn't use */}
        {isGrag && data.retriever && (
          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginBottom: 12 }}>
            {[data.retriever, 'CITES traversal', `${data.context_tokens ?? 0} ctx tokens`].map(tag => (
              <span key={tag} style={{
                background: 'rgba(22,163,74,0.1)', color: '#16a34a',
                borderRadius: 6, padding: '2px 8px', fontSize: 10, fontWeight: 600,
                border: '1px solid rgba(22,163,74,0.2)',
              }}>{tag}</span>
            ))}
          </div>
        )}

        {/* Answer toggle */}
        <button
          onClick={() => setExpanded(x => !x)}
          style={{
            background: 'none', border: `1px solid ${t.border}`, cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            width: '100%', padding: '8px 12px', borderRadius: 8,
            fontSize: 12, fontWeight: 600, color: t.textMuted,
            marginBottom: expanded ? 8 : 0,
          }}
        >
          <span>Answer</span>
          <ChevronDown size={13} style={{ transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
        </button>
        {expanded && (
          <div style={{
            background: t.metricBg, border: `1px solid ${t.border}`,
            borderRadius: 10, padding: '12px 14px',
            fontSize: 13, color: t.textMuted, lineHeight: 1.75,
            maxHeight: 160, overflowY: 'auto',
          }}>
            {data.answer || <em style={{ color: t.textSubtle }}>No answer returned.</em>}
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Results Section ─── */
function ResultsSection({ result, t }) {
  const pct    = result.token_reduction_pct;
  const graphFailed  = result.graphrag_status && result.graphrag_status !== 'ok';
  const basicRagFailed = result.basic_rag?.status === 'faiss_unavailable';
  const maxTok = Math.max(...PIPELINE_KEYS.map(k => result[k].total_tokens));
  const chartData = PIPELINE_KEYS.map(k => ({
    name: PIPELINE_LABELS[k], tokens: result[k].total_tokens, color: PIPELINE_COLORS[k],
  }));

  return (
    <div className="animate-fade-up">
      {/* GraphRAG failure notice — only shown when GraphRAG itself failed */}
      {graphFailed && (
        <div style={{
          background: t.errorBg, border: `1px solid ${t.errorBorder}`,
          borderRadius: 20, padding: '24px 28px', marginBottom: 20,
          display: 'flex', alignItems: 'flex-start', gap: 16,
        }}>
          <XCircle size={28} color={t.errorText} style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div style={{ fontSize: 18, fontWeight: 800, color: t.errorText, marginBottom: 6 }}>
              {result.graphrag_status === 'tg_unavailable'
                ? 'TigerGraph knowledge-graph service is offline'
                : 'GraphRAG found no matching entities in the knowledge graph'}
            </div>
            <div style={{ fontSize: 14, color: t.textMuted, lineHeight: 1.6 }}>
              {result.graphrag_status === 'tg_unavailable' ? (
                <>
                  The graph backend didn't respond, so no context could be retrieved — and therefore{' '}
                  <strong>no token reduction is reported</strong> (rather than a fake percentage).
                  Resume the TigerGraph instance and try again.
                </>
              ) : (
                <>
                  This question references entities that aren't in the knowledge graph, so there's
                  no genuine context to retrieve — and therefore <strong>no token reduction to report</strong>.
                  Try one of the <strong style={{ color: '#16a34a' }}>Featured Questions</strong> above,
                  which are verified to exist in the graph.
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Basic RAG unavailable notice — separate from GraphRAG status */}
      {!graphFailed && basicRagFailed && (
        <div style={{
          background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.35)',
          borderRadius: 20, padding: '18px 24px', marginBottom: 20,
          display: 'flex', alignItems: 'flex-start', gap: 14,
        }}>
          <Database size={22} color="#f97316" style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#f97316', marginBottom: 4 }}>
              Basic RAG unavailable — FAISS index not on this server
            </div>
            <div style={{ fontSize: 13, color: t.textMuted, lineHeight: 1.6 }}>
              GraphRAG retrieved context successfully and answered your question.
              Token reduction vs Basic RAG cannot be shown because the FAISS vector index isn't deployed on this server.
            </div>
          </div>
        </div>
      )}

      {/* Token Reduction Hero — only when GraphRAG retrieved AND Basic RAG is available to compare */}
      {!graphFailed && !basicRagFailed && (
      <div style={{
        background: 'linear-gradient(135deg, #052e16 0%, #14532d 50%, #166534 100%)',
        borderRadius: 20, padding: '32px 36px', marginBottom: 20,
        border: '1px solid rgba(34,197,94,0.3)',
        boxShadow: '0 8px 32px rgba(22,163,74,0.2)',
        display: 'flex', alignItems: 'center', gap: 40, flexWrap: 'wrap',
        position: 'relative', overflow: 'hidden',
      }}>
        <div style={{ position: 'absolute', top: -40, right: -40, width: 160, height: 160, borderRadius: '50%', background: 'rgba(34,197,94,0.05)' }} />
        <div style={{ position: 'absolute', bottom: -20, right: 80, width: 80, height: 80, borderRadius: '50%', background: 'rgba(34,197,94,0.08)' }} />

        {/* Token reduction stat */}
        <div style={{ textAlign: 'center', position: 'relative' }}>
          <div style={{ fontSize: 11, color: 'rgba(74,222,128,0.8)', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 4 }}>
            Token Reduction
          </div>
          <div style={{ fontSize: 64, fontWeight: 900, lineHeight: 1, color: '#4ade80', letterSpacing: '-2px' }}>
            <AnimatedNumber value={pct} suffix="%" />
          </div>
          <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.5)', marginTop: 4 }}>GraphRAG vs Basic RAG</div>
        </div>

        <div style={{ width: 1, height: 80, background: 'rgba(255,255,255,0.1)' }} />

        {/* Token bars */}
        <div style={{ flex: 1, minWidth: 240 }}>
          {PIPELINE_KEYS.map(k => (
            <TokenBar
              key={k}
              label={PIPELINE_LABELS[k]}
              tokens={result[k].total_tokens}
              maxTokens={maxTok}
              color={PIPELINE_COLORS[k]}
              t={{ ...t, border: 'rgba(255,255,255,0.1)', textMuted: 'rgba(255,255,255,0.6)' }}
            />
          ))}
        </div>

        <div style={{ width: 1, height: 80, background: 'rgba(255,255,255,0.1)' }} />

        {/* Cost reduction stat */}
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: 'rgba(74,222,128,0.8)', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 4 }}>
            Cost Saved
          </div>
          <div style={{ fontSize: 36, fontWeight: 900, color: '#86efac', letterSpacing: '-1px' }}>
            {result.cost_reduction_pct > 0 ? '-' : '+'}<AnimatedNumber value={Math.abs(result.cost_reduction_pct)} suffix="%" />
          </div>
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', marginTop: 4 }}>per query vs Basic RAG</div>
        </div>
      </div>
      )}

      {/* Chart + Pipeline cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 16, marginBottom: 20 }}>
        {/* Bar chart */}
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`,
          borderRadius: 16, padding: '20px', boxShadow: t.cardShadow,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: t.textSubtle, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 16 }}>
            Token Comparison
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={t.chartGrid} vertical={false} />
              <XAxis dataKey="name" stroke={t.textSubtle} tick={{ fill: t.textMuted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis stroke={t.textSubtle} tick={{ fill: t.textMuted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: t.tooltipBg, border: `1px solid ${t.border}`, borderRadius: 10, boxShadow: t.cardShadow, fontSize: 13 }}
                labelStyle={{ color: t.text, fontWeight: 700 }}
                itemStyle={{ color: t.textMuted }}
                cursor={{ fill: t.accentGlow }}
              />
              <Bar dataKey="tokens" radius={[8, 8, 0, 0]} maxBarSize={56}>
                {chartData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

        </div>

        {/* Pipeline cards — driven by PIPELINE_KEYS registry */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {PIPELINE_KEYS.map(key => (
            <PipelineCard
              key={key} name={key} data={result[key]}
              judge={result[`judge_${key}`]}
              judgeSource={result[`judge_${key}_source`]}
              t={t}
            />
          ))}
        </div>
      </div>

      {/* BERTScore */}
      {result.bertscore && result.bertscore.raw_f1 > 0 && (
        <div style={{
          background: t.bertBg, border: `1px solid ${t.bertBorder}`,
          borderRadius: 16, padding: '18px 24px', marginBottom: 20,
          display: 'flex', alignItems: 'center', gap: 28, flexWrap: 'wrap',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Award size={18} color={t.bertText} />
            <span style={{ fontWeight: 700, fontSize: 14, color: t.bertText }}>BERTScore</span>
          </div>
          {[
            { label: 'Raw F1',      value: result.bertscore.raw_f1 },
            { label: 'Rescaled F1', value: result.bertscore.rescaled_f1 },
          ].map(({ label, value }) => (
            <div key={label}>
              <div style={{ fontSize: 10, color: t.bertText, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2, fontWeight: 600 }}>{label}</div>
              <div style={{ fontSize: 24, fontWeight: 800, color: t.bertText }}>{value}</div>
            </div>
          ))}
          <div style={{ marginLeft: 'auto' }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              background: result.bertscore.bonus_hit ? t.badgePassBg : t.badgeFailBg,
              color: result.bertscore.bonus_hit ? t.badgePassText : t.badgeFailText,
              borderRadius: 20, padding: '5px 14px', fontSize: 12, fontWeight: 700,
            }}>
              {result.bertscore.bonus_hit ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
              {result.bertscore.bonus_hit ? 'BONUS HIT' : 'BONUS MISSED'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Main App ─── */
export default function App() {
  const [question,  setQuestion]  = useState('');
  const [groundTruth, setGT]      = useState('');
  const [loading,   setLoading]   = useState(false);
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState('');
  const [history,   setHistory]   = useState([]);
  const [darkMode,  setDarkMode]  = useState(false);
  const [selDomain, setDomain]    = useState('');
  const [showHist,  setShowHist]  = useState(false);
  const inputRef = useRef();

  const t          = THEMES[darkMode ? 'dark' : 'light'];
  const domainData = DOMAIN_QUESTIONS.find(d => d.domain === selDomain);

  async function handleRun(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true); setError(''); setResult(null);
    try {
      const { data } = await axios.post(`${API_BASE}/compare`, {
        question:     question.trim(),
        ground_truth: groundTruth.trim(),
      });
      setResult(data);
      setHistory(prev => [{
        question:        question.trim(),
        graphrag_tokens: data.graphrag.total_tokens,
        reduction_pct:   data.token_reduction_pct,
        judge:           data.judge_graphrag || '—',
        bertscore:       data.bertscore?.raw_f1 ?? '—',
      }, ...prev].slice(0, 20));
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  function selectQuestion(q, a = '') {
    setQuestion(q); setGT(a); setResult(null); setError('');
    setTimeout(() => inputRef.current?.focus(), 100);
  }

  const inputStyle = {
    background: t.inputBg, border: `1.5px solid ${t.border}`,
    borderRadius: 14, padding: '16px 20px', fontSize: 16,
    color: t.text, outline: 'none', width: '100%',
    transition: 'border-color 0.15s, box-shadow 0.15s',
    boxSizing: 'border-box', fontFamily: 'inherit',
  };

  return (
    <div style={{
      minHeight: '100vh', background: t.pageBg, color: t.text,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      transition: 'background 0.3s',
    }}>

      {/* ═══ HERO ═══ */}
      <div style={{ background: t.heroBg, padding: '0 24px 0', position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: -80, left: -80, width: 300, height: 300, borderRadius: '50%', background: 'rgba(22,163,74,0.08)', filter: 'blur(40px)' }} />
        <div style={{ position: 'absolute', top: -40, right: 100, width: 200, height: 200, borderRadius: '50%', background: 'rgba(59,130,246,0.06)', filter: 'blur(30px)' }} />

        <div style={{ maxWidth: 1280, margin: '0 auto' }}>
          {/* Nav */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '20px 0 0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#4ade80', boxShadow: '0 0 8px #4ade80', animation: 'pulse-ring 2s ease infinite' }} />
              <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.6)', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                TigerGraph GraphRAG · Legal Citation Graph
              </span>
            </div>
            <button
              onClick={() => setDarkMode(d => !d)}
              style={{
                background: t.toggleBg, color: '#fff',
                border: '1px solid rgba(255,255,255,0.2)', borderRadius: 10,
                padding: '8px 16px', fontSize: 13, fontWeight: 600,
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
                backdropFilter: 'blur(10px)',
              }}
            >
              {darkMode ? <Sun size={14} /> : <Moon size={14} />}
              {darkMode ? 'Light' : 'Dark'}
            </button>
          </div>

          {/* Hero text */}
          <div style={{ padding: '60px 0 52px', textAlign: 'center' }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              background: 'rgba(22,163,74,0.15)', border: '1px solid rgba(22,163,74,0.35)',
              borderRadius: 24, padding: '8px 22px', marginBottom: 28,
            }}>
              <Network size={15} color="#4ade80" />
              <span style={{ fontSize: 14, color: '#4ade80', fontWeight: 700, letterSpacing: '0.02em' }}>
                117.5M Token Legal Citation Graph
              </span>
            </div>
            <h1 style={{
              fontSize: 'clamp(36px, 6vw, 68px)', fontWeight: 900, color: '#ffffff',
              margin: '0 0 20px', letterSpacing: '-2px', lineHeight: 1.05,
            }}>
              GraphRAG Pipeline{' '}
              <span style={{ background: 'linear-gradient(135deg, #4ade80, #22c55e)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Comparison
              </span>
            </h1>
            <p style={{
              fontSize: 20, color: 'rgba(255,255,255,0.65)', margin: '0 0 44px',
              maxWidth: 600, marginLeft: 'auto', marginRight: 'auto', lineHeight: 1.65, fontWeight: 400,
            }}>
              Run any question through <strong style={{ color: '#4ade80', fontWeight: 700 }}>GraphRAG</strong>,{' '}
              <strong style={{ color: '#4ade80', fontWeight: 700 }}>Basic RAG</strong>, and{' '}
              <strong style={{ color: '#4ade80', fontWeight: 700 }}>LLM-Only</strong> side by side —
              see the tokens, latency, and accuracy for yourself
            </p>

            {/* Stat cards */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: 16, flexWrap: 'wrap' }}>
              {[
                { icon: BookOpen, value: '63,632',     label: 'Court Opinions' },
                { icon: Hash,     value: '117.5M',     label: 'Tokens Indexed' },
                { icon: Layers,   value: '500,959',    label: 'FAISS Chunks' },
                { icon: GitMerge, value: '9,632',      label: 'Citation Edges' },
              ].map(({ icon: Icon, value, label }) => (
                <div key={label} style={{
                  background: t.statCard, border: `1px solid ${t.statCardBorder}`,
                  borderRadius: 18, padding: '20px 28px', minWidth: 150,
                  backdropFilter: 'blur(10px)', textAlign: 'center',
                }}>
                  <Icon size={20} color="#4ade80" style={{ marginBottom: 10, display: 'block', margin: '0 auto 10px' }} />
                  <div style={{ fontSize: 22, fontWeight: 900, color: '#fff', letterSpacing: '-0.5px' }}>{value}</div>
                  <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.55)', marginTop: 4, fontWeight: 500 }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ═══ MAIN CONTENT ═══ */}
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '36px 32px 80px' }}>

        {/* ── Featured Questions ── */}
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`,
          borderRadius: 24, padding: '32px 36px', marginBottom: 24,
          boxShadow: t.cardShadow,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ background: 'rgba(22,163,74,0.1)', borderRadius: 10, padding: 10 }}>
                <Sparkles size={20} color="#16a34a" />
              </div>
              <div>
                <div style={{ fontSize: 18, fontWeight: 800, color: t.text }}>Featured Questions</div>
                <div style={{ fontSize: 13, color: t.textSubtle, marginTop: 2 }}>Verified working questions with highest token reduction</div>
              </div>
            </div>
            <span style={{
              background: 'rgba(22,163,74,0.1)', color: '#16a34a',
              borderRadius: 20, padding: '4px 12px', fontSize: 11, fontWeight: 700,
              border: '1px solid rgba(22,163,74,0.2)',
            }}>Best Reduction</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 14, marginBottom: 28 }}>
            {FEATURED_QUESTIONS.map(sq => (
              <button
                key={sq.question}
                onClick={() => selectQuestion(sq.question, sq.answer)}
                style={{
                  background: t.surface2, border: `1.5px solid ${t.border}`,
                  borderRadius: 16, padding: '20px 22px', textAlign: 'left',
                  cursor: 'pointer', transition: 'all 0.2s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = '#16a34a';
                  e.currentTarget.style.background = 'rgba(22,163,74,0.05)';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(22,163,74,0.12)';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = t.border;
                  e.currentTarget.style.background = t.surface2;
                  e.currentTarget.style.transform = 'none';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 20 }}>{sq.icon}</span>
                    <span style={{ fontSize: 11, fontWeight: 700, color: '#16a34a', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{sq.label}</span>
                  </div>
                </div>
                <div style={{ fontSize: 15, color: t.text, lineHeight: 1.55, fontWeight: 600, marginBottom: 10 }}>{sq.question}</div>
                <div style={{ fontSize: 12, color: t.textSubtle, display: 'flex', alignItems: 'center', gap: 5 }}>
                  <TrendingDown size={12} /> {sq.hint}
                </div>
              </button>
            ))}
          </div>

          {/* Browse by Domain */}
          <div style={{ borderTop: `1px solid ${t.border}`, paddingTop: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
              <div style={{ background: 'rgba(22,163,74,0.1)', borderRadius: 8, padding: '7px 9px', display: 'flex', alignItems: 'center' }}>
                <BookOpen size={16} color="#16a34a" />
              </div>
              <div>
                <div style={{ fontSize: 20, fontWeight: 800, color: t.text }}>Browse by Domain</div>
                <div style={{ fontSize: 14, color: t.textSubtle, marginTop: 2 }}>Try more questions from different topics</div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-start' }}>
              <select
                value={selDomain}
                onChange={e => setDomain(e.target.value)}
                style={{
                  background: t.inputBg, border: `1.5px solid ${t.border}`,
                  borderRadius: 12, padding: '12px 18px', fontSize: 14,
                  color: selDomain ? t.text : t.textMuted,
                  cursor: 'pointer', outline: 'none', minWidth: 260,
                  fontFamily: 'inherit', fontWeight: 500,
                }}
              >
                <option value="">Select a domain…</option>
                {DOMAIN_QUESTIONS.map(d => (
                  <option key={d.domain} value={d.domain}>{d.domain}</option>
                ))}
              </select>

              {domainData && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, flex: 1 }}>
                  {domainData.questions.map(item => (
                    <button
                      key={item.question}
                      onClick={() => selectQuestion(item.question, item.answer)}
                      style={{
                        background: t.surface2, border: `1px solid ${t.border}`,
                        borderRadius: 22, padding: '8px 18px', fontSize: 13,
                        color: t.text, cursor: 'pointer', transition: 'all 0.15s',
                        fontFamily: 'inherit', fontWeight: 500,
                      }}
                      onMouseEnter={e => {
                        e.currentTarget.style.borderColor = '#16a34a';
                        e.currentTarget.style.color = '#16a34a';
                        e.currentTarget.style.background = 'rgba(22,163,74,0.06)';
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.borderColor = t.border;
                        e.currentTarget.style.color = t.text;
                        e.currentTarget.style.background = t.surface2;
                      }}
                    >
                      {item.question}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── Query Form ── */}
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`,
          borderRadius: 24, padding: '32px 36px', marginBottom: 28,
          boxShadow: t.cardShadow,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 22 }}>
            <div style={{ background: 'rgba(22,163,74,0.1)', borderRadius: 10, padding: 10 }}>
              <Zap size={20} color="#16a34a" />
            </div>
            <div>
              <div style={{ fontSize: 18, fontWeight: 800, color: t.text }}>Ask a Question</div>
              <div style={{ fontSize: 13, color: t.textSubtle, marginTop: 2 }}>
                Runs LLM-Only · Basic RAG · GraphRAG in parallel
              </div>
            </div>
          </div>
          <form onSubmit={handleRun} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <input
              ref={inputRef}
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="e.g. How did People v. Contes apply the harmless error principle?"
              style={inputStyle}
              onFocus={e => { e.target.style.borderColor = '#16a34a'; e.target.style.boxShadow = '0 0 0 3px rgba(22,163,74,0.1)'; }}
              onBlur={e => { e.target.style.borderColor = t.border; e.target.style.boxShadow = 'none'; }}
            />
            <input
              value={groundTruth}
              onChange={e => setGT(e.target.value)}
              placeholder="Ground truth (optional) — enables LLM Judge + BERTScore evaluation"
              style={{ ...inputStyle, color: t.textMuted }}
              onFocus={e => { e.target.style.borderColor = '#16a34a'; e.target.style.boxShadow = '0 0 0 3px rgba(22,163,74,0.1)'; }}
              onBlur={e => { e.target.style.borderColor = t.border; e.target.style.boxShadow = 'none'; }}
            />
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <button
                type="submit"
                disabled={loading || !question.trim()}
                style={{
                  background: (loading || !question.trim()) ? t.btnDisabledBg : 'linear-gradient(135deg, #16a34a, #15803d)',
                  color: (loading || !question.trim()) ? t.btnDisabledText : '#fff',
                  border: 'none', borderRadius: 14, padding: '16px 40px',
                  fontSize: 16, fontWeight: 700, cursor: (loading || !question.trim()) ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', gap: 10,
                  boxShadow: (loading || !question.trim()) ? 'none' : '0 6px 20px rgba(22,163,74,0.4)',
                  transition: 'all 0.2s', fontFamily: 'inherit',
                }}
              >
                <BarChart2 size={18} />
                {loading ? 'Running all 3 pipelines…' : 'Run All 3 Pipelines'}
              </button>
              {(question || result) && !loading && (
                <button
                  type="button"
                  onClick={() => { setQuestion(''); setGT(''); setResult(null); setError(''); }}
                  style={{
                    background: 'none', border: `1px solid ${t.border}`,
                    borderRadius: 14, padding: '16px 24px',
                    fontSize: 15, color: t.textMuted, cursor: 'pointer', fontFamily: 'inherit',
                  }}
                >
                  Clear
                </button>
              )}
              {result && !loading && result.token_reduction_pct != null && (
                <div style={{
                  marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8,
                  background: 'rgba(22,163,74,0.1)', border: '1px solid rgba(22,163,74,0.2)',
                  borderRadius: 10, padding: '8px 14px',
                }}>
                  <TrendingDown size={14} color="#16a34a" />
                  <span style={{ fontSize: 13, fontWeight: 700, color: '#16a34a' }}>
                    {result.token_reduction_pct}% token reduction achieved
                  </span>
                </div>
              )}
              {result && !loading && result.token_reduction_pct == null && result.graphrag?.status === 'ok' && (
                <div style={{
                  marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8,
                  background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.35)',
                  borderRadius: 10, padding: '8px 14px',
                }}>
                  <Database size={14} color="#f97316" />
                  <span style={{ fontSize: 13, fontWeight: 700, color: '#f97316' }}>
                    Basic RAG unavailable — token reduction not computed
                  </span>
                </div>
              )}
              {result && !loading && result.graphrag_status && result.graphrag_status !== 'ok' && (
                <div style={{
                  marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8,
                  background: t.errorBg, border: `1px solid ${t.errorBorder}`,
                  borderRadius: 10, padding: '8px 14px',
                }}>
                  <XCircle size={14} color={t.errorText} />
                  <span style={{ fontSize: 13, fontWeight: 700, color: t.errorText }}>
                    No graph context — not in knowledge graph
                  </span>
                </div>
              )}
            </div>
          </form>
        </div>

        {/* ── Error ── */}
        {error && (
          <div style={{
            background: t.errorBg, border: `1px solid ${t.errorBorder}`,
            borderRadius: 14, padding: '14px 18px', marginBottom: 20,
            color: t.errorText, fontSize: 14, display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <XCircle size={16} /> {error}
          </div>
        )}

        {/* ── Loading ── */}
        {loading && (
          <div style={{
            background: t.surface, border: `1px solid ${t.border}`,
            borderRadius: 20, marginBottom: 20, boxShadow: t.cardShadow,
          }}>
            <Spinner t={t} />
          </div>
        )}

        {/* ── Results ── */}
        {result && !loading && <ResultsSection result={result} t={t} />}

        {/* ── History ── */}
        {history.length > 0 && (
          <div style={{
            background: t.surface, border: `1px solid ${t.border}`,
            borderRadius: 20, boxShadow: t.cardShadow, overflow: 'hidden',
          }}>
            <button
              onClick={() => setShowHist(h => !h)}
              style={{
                width: '100%', padding: '18px 24px',
                background: 'none', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 10,
                borderBottom: showHist ? `1px solid ${t.border}` : 'none',
              }}
            >
              <History size={15} color={t.textMuted} />
              <span style={{ fontSize: 14, fontWeight: 700, color: t.text }}>Query History</span>
              <span style={{
                background: t.surface2, border: `1px solid ${t.border}`,
                borderRadius: 20, padding: '1px 9px', fontSize: 11, color: t.textMuted, fontWeight: 600,
              }}>{history.length}</span>
              <ChevronDown size={14} color={t.textSubtle} style={{ marginLeft: 'auto', transform: showHist ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </button>
            {showHist && (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: t.surface2 }}>
                      {['Question', 'GraphRAG Tokens', 'Reduction', 'Judge', 'BERTScore'].map(h => (
                        <th key={h} style={{
                          padding: '10px 16px', borderBottom: `1px solid ${t.border}`,
                          color: t.textSubtle, textAlign: 'left', fontWeight: 700,
                          fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.07em',
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((row, i) => (
                      <tr
                        key={i}
                        style={{ borderBottom: `1px solid ${t.border}`, cursor: 'pointer' }}
                        onClick={() => selectQuestion(row.question)}
                        onMouseEnter={e => e.currentTarget.style.background = t.surfaceHover}
                        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                      >
                        <td style={{ padding: '11px 16px', color: t.text, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.question}</td>
                        <td style={{ padding: '11px 16px', color: PIPELINE_COLORS.graphrag, fontWeight: 700 }}>{row.graphrag_tokens.toLocaleString()}</td>
                        <td style={{ padding: '11px 16px', fontWeight: 800, color: row.reduction_pct == null ? t.textSubtle : t.reductionStat }}>
                          {row.reduction_pct == null ? 'no match' : `${row.reduction_pct > 0 ? '-' : '+'}${Math.abs(row.reduction_pct)}%`}
                        </td>
                        <td style={{ padding: '11px 16px' }}>
                          {row.judge !== '—' ? <JudgeBadge judge={row.judge} t={t} /> : <span style={{ color: t.textSubtle }}>—</span>}
                        </td>
                        <td style={{ padding: '11px 16px', color: t.textMuted, fontWeight: 600 }}>{row.bertscore}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ── Footer ── */}
        <div style={{ marginTop: 48, textAlign: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 8 }}>
            <Network size={14} color="#16a34a" />
            <span style={{ fontSize: 13, fontWeight: 700, color: t.textMuted }}>
              TigerGraph GraphRAG Pipeline Comparison
            </span>
          </div>
          <div style={{ fontSize: 12, color: t.textSubtle, lineHeight: 1.6 }}>
            Powered by <strong style={{ color: t.textMuted }}>Gemini 2.5 Flash</strong> ·{' '}
            <strong style={{ color: t.textMuted }}>TigerGraph</strong> Knowledge Graph ·{' '}
            <strong style={{ color: t.textMuted }}>FAISS</strong> Vector Index ·{' '}
            <strong style={{ color: t.textMuted }}>102.9M</strong> tokens indexed
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse-ring {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.5; transform: scale(1.3); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-up { animation: fadeUp 0.4s ease forwards; }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
      `}</style>
    </div>
  );
}
