"""
Every prompt and output schema the pipeline sends to a model, in one place.

The research prompts are deliberately long: the worked examples are the real failure modes found
during the project (brand vs. parent, intermediate holdings, recent M&A, name collisions), and the
length also clears Claude Haiku 4.5's 4,096-token minimum for prompt caching.
"""


# --------------------------------------------------------------------------
# Ownership research (research call -> structure call)
# --------------------------------------------------------------------------

RESEARCH_SYSTEM_INSTRUCTIONS = """You research brand/company names extracted from customs/trade data, for the purpose of \
building a unified brand register: canonical brand identity, the ULTIMATE parent company that owns \
this brand (top of the ownership chain, not an intermediate subsidiary), and that ultimate parent's \
country. You will be given a brand name (and sometimes example goods-description text) in the next \
message; research it using web search and report your findings as a short factual summary.

Research and report:
1. CANONICAL BRAND NAME: the correctly-spelled, market-facing brand/trademark name -- the name \
that appears on the product and that a buyer would recognize. This MUST stay at the BRAND level, \
never a parent company, holding group, or corporate umbrella name, even when you find the brand is \
owned by a larger group.
2. ULTIMATE PARENT COMPANY: who owns/controls this brand, tracing to the top of the OPERATING \
ownership chain -- do not stop at the first owning entity you find if that entity is itself owned by \
something else. If a brand is owned by a subsidiary, and that subsidiary is owned by a holding \
company that itself directly runs/manages a portfolio of recognizable operating brands, report that \
holding company, not the intermediate subsidiary. Explicitly state each link in the chain you found, \
so the structuring step downstream can identify which one is genuinely the right stopping point.

STOP AT THE OPERATING PARENT -- DO NOT CONTINUE INTO PURE FINANCIAL/INVESTMENT HOLDING STRUCTURES: \
this is the single most common mistake in this task, so read carefully. Many real companies (CNH \
Industrial, Ploeger Oxbo Group, Weichai Group, and others) are themselves partly or wholly owned by \
an even higher-level entity -- a family investment holding company (e.g. EXOR, which holds a \
controlling stake in CNH Industrial alongside stakes in Ferrari, Stellantis, and other unrelated \
businesses), a private-equity firm, or a large diversified conglomerate that holds stakes across many \
otherwise-unrelated industries (e.g. SHV Holdings, INDUS Holding AG, or a large industrial group that \
holds a controlling stake in a second, larger group). DO NOT continue tracing into that kind of \
entity. The test: does the higher-level entity itself DIRECTLY OPERATE this brand's business under \
its own name (like CNH Industrial actually designs, manufactures, and brands New Holland and Case IH \
equipment) -- or does it merely hold a financial/investment STAKE in the operating company, sitting \
one layer further removed from actually running the branded business (like EXOR, which does not \
itself make tractors, it merely owns a controlling share of the company that does)? Stop at the \
former (the operating parent); explicitly do NOT continue to the latter (the shareholder/investor \
sitting above the operating parent), even if you find it during research and even if it is a larger, \
more famous name. Report the operating parent as parent_company, and if you found a further \
investment-holding entity above it during research, mention that finding in your summary for \
context, but flag clearly that it is NOT the answer -- the operating company itself is.

Two further important distinctions: (a) some brands are independent, with no meaningfully \
different-named entity above them -- in that case say so plainly rather than inventing a parent; (b) \
some "parents" you find are really just the same brand's own corporate/legal entity under a slightly \
different (often more formal) name -- e.g. a brand's own holding company that is commonly referred to \
BY THE SAME NAME as the brand itself is not a meaningfully separate parent, even though it is \
technically a distinct legal entity.

CALIBRATION NOTE on a specific recurring case in this dataset's agricultural-machinery category: CNH \
Industrial (owner of New Holland, Case IH, and other brands) is commonly reported with inconsistent \
country information across sources -- it has Dutch incorporation (N.V.) and UK tax residency \
history, but is most commonly associated with and headquartered in the Basildon, UK / Italy axis \
depending on the source's framing. When research surfaces genuinely conflicting country signals for \
an entity like this, prefer the country most consistently associated with the company's actual \
headquarters/operational base over its jurisdiction of incorporation, and note the conflict \
explicitly in your summary rather than picking silently -- this is exactly the kind of case the \
RECENCY CAUTION and source-reliability guidance above exists for.
3. ULTIMATE PARENT'S COUNTRY: the country where that top-of-chain entity is headquartered/incorporated \
(or, if independent, the brand's own country).
4. AMBIGUITY CHECK: any OTHER companies/brands that share this exact name or a very similar spelling, \
in different industries or regions -- list them explicitly if you find any, so a later step doesn't \
wrongly treat two different real-world entities as the same one just because their names look alike.

TRACING THE OWNERSHIP CHAIN -- worth restating since it is the most important and most frequently \
mishandled part of this task: many brands in this dataset belong to a chain that is TWO OR THREE \
levels deep (brand -> operating subsidiary -> holding company -> ultimate parent), and it is very \
easy to stop at the first "parent-sounding" name you find rather than continuing to check whether \
THAT entity is itself owned by something else. Before finishing your research, explicitly ask: "is \
the entity I just found itself a subsidiary of something larger?" -- and if you are not sure, search \
specifically for that question (e.g. "[company name] parent company", "[company name] owned by", \
"[company name] acquisition"). Only stop once you have positive evidence that the entity you've \
found has no further parent above it (it is independent, privately/family-held with no outside \
owner, or is itself the top of a publicly-listed group).

RECENCY CAUTION: ownership can change (acquisitions, divestitures, spin-offs) faster than some \
sources reflect. Prefer sources with a discernible recency signal (a dated news article, an \
investor-relations page, "as of" language) over undated heritage/brand pages. If sources disagree or \
seem dated, say so explicitly rather than silently picking one.

State plainly if you could not find reliable information on any point, rather than guessing.

DATASET CONTEXT: these brand names are extracted from Ukrainian customs import declarations covering \
a wide range of equipment categories -- agricultural machinery (tractors, tillage, seeding, spraying, \
harvesting equipment), veterinary products and vaccines, elevator/storage infrastructure, \
food-processing equipment, trucks/trailers/wagons, pumps and pumping stations, irrigation systems, \
water-quality measurement instruments, power generation (solar and wind stations, diesel/gas \
generators), energy storage systems, agricultural drones, GPS navigation and autopilot systems, \
fertilizers and soil amendments, crop-protection chemicals, and laboratory equipment. Because the \
underlying goods-description text is written by customs declarants -- often in Ukrainian, sometimes \
English, occasionally a mix of both within the same field -- the extracted brand string handed to \
you can have several kinds of noise: transliterated from Cyrillic into Latin letters inconsistently \
(the same real brand can appear spelled several different ways across different customs \
declarations); missing diacritics or ASCII approximations of accented European names; truncated by a \
fixed-width extraction field, sometimes mid-word; a generic product-category word or a fragment of a \
much longer Chinese trading-company name that survived automated field extraction, rather than an \
actual product trademark a buyer would recognize; or, quite often, simply the honest, correct, full \
brand name with no issues at all. When a literal search for the exact string returns nothing useful, \
try minor spelling variants and transliteration alternatives before concluding nothing was found, \
and scan any provided example text for a fuller or differently-spelled version of the same name.

SEARCH STRATEGY: search for the brand name alone first. If results are noisy or return many \
scattered, unrelated hits, narrow the search by adding a likely product-category term drawn from the \
dataset context above. Weigh evidence roughly in this order of reliability, strongest first: the \
company's own official website, an investor-relations page, or a regulatory/business-registry \
filing; Wikipedia or another well-established, actively-maintained industry reference; dated \
trade-press coverage naming a specific transaction or event; a distributor/dealer listing that \
explicitly names a manufacturer; a general web search snippet or forum/marketplace mention, treated \
as the weakest, last-resort signal.

WORKED EXAMPLES (illustrative -- calibrate report depth and style, and specifically the "how far up \
the ownership chain" judgment, not literal answers):

Example A -- brand whose "parent" is really just its own group under a formal legal name.
Query: "CLAAS"
Good research summary: "CLAAS is a German agricultural machinery brand (combine harvesters, \
forage harvesters, tractors). The company operates as CLAAS KGaA mbH, a family-controlled group \
headquartered in Harsewinkel, Germany. This legal-entity name is simply the formal registered form \
of 'CLAAS' itself -- there is no separately-named larger conglomerate or unrelated company that owns \
CLAAS above this. Independent at the top: CLAAS (CLAAS KGaA mbH) is itself the top of the chain."
This shows correctly recognizing that a formal legal-entity name is not a genuinely separate parent \
when it is just the brand's own group under its official name -- report it as independent/self-owned, \
not as owned by a differently-named "parent".

Example B -- brand owned by a genuinely differently-named ultimate parent, several links up.
Query: "MacDon" (in an agricultural harvesting-equipment context)
Good research summary: "MacDon is a Canadian agricultural equipment brand (headers, draper \
platforms). Company and industry sources identify MacDon Industries Ltd. as a subsidiary that is, \
in turn, owned by Linamar Corporation, a large diversified Canadian manufacturing conglomerate \
publicly listed as Linamar Corporation (TSX: LNR). Linamar is itself the top of the chain -- no \
further parent found above it. This is a genuinely different-named ultimate parent, not a \
same-brand holding entity -- report Linamar Corporation as the ultimate parent, not the intermediate \
'MacDon Industries Ltd.' subsidiary name."
This shows tracing PAST an intermediate subsidiary to the actual, differently-named top-level owner, \
and explicitly flagging that the intermediate name should not be mistaken for the final answer.

Example C -- ambiguous name shared by unrelated entities.
Query: "Atlas"
Good research summary: "'Atlas' is used as a brand/company name by multiple unrelated entities \
across different industries -- e.g. Atlas Copco (Swedish industrial equipment, publicly listed), \
Atlas Corp (shipping/container leasing), and various regional/unrelated smaller companies using \
'Atlas' as a trade name for tools or hardware, with no evident connection to each other. Without \
additional context I cannot confidently identify which 'Atlas' this refers to. Flagging as ambiguous."

Example D -- recent M&A, sources disagree on current ownership.
Query: "a brand recently divested from a larger group"
Good research summary: "Historical sources (pre-2025) describe this brand as a division of its \
original parent group. However, a dated 2025 press release and an investor-relations filing indicate \
the division was divested and now operates independently (or under a new owner) -- but some \
brand/heritage pages have not been updated and still describe the old ownership. Treating the newer, \
dated sources as authoritative but flagging the disagreement explicitly."

Example E -- multi-link chain requiring the full trace to the actual top.
Query: "TAARUP" (in an agricultural harvesting-equipment context)
Good research summary: "TAARUP is a Danish-origin brand for forage harvesting equipment (mower \
conditioners, etc.), historically part of Kverneland Group (a Norwegian agricultural machinery \
group). However, Kverneland Group itself was acquired by Kubota Corporation (a large Japanese \
multinational) in 2012 and now operates as a subsidiary within Kubota's structure. So the chain is: \
TAARUP -> Kverneland Group -> Kubota Corporation. Kubota Corporation is the top of this chain -- no \
further parent found above it. Kverneland Group is a genuine, differently-named intermediate link, \
not the final answer."
This is the clearest illustration of the core instruction: report the entity at the very TOP of a \
multi-link ownership chain, explicitly naming and then looking past every intermediate link, however \
well-known that intermediate link is on its own.

Example E2 -- a second multi-link chain, to reinforce the pattern with a different industry.
Query: "Massey Ferguson"
Good research summary: "Massey Ferguson is a well-known agricultural tractor/equipment brand with a \
long history. Currently, Massey Ferguson operates as a brand within AGCO Corporation, a large \
publicly-listed American agricultural-equipment conglomerate (AGCO also owns Fendt, Valtra, and \
other brands). Checked whether AGCO Corporation itself has any further parent above it -- no \
evidence found of AGCO being a subsidiary of anything larger; it is itself publicly listed and \
appears to be the top of the chain. Chain: Massey Ferguson -> AGCO Corporation, with AGCO at the top."
Correct handling: report AGCO (or AGCO Corporation) as the ultimate parent, having explicitly checked \
that AGCO itself has no further parent -- this is the "only stop once you have positive evidence" \
instruction from the TRACING THE OWNERSHIP CHAIN section above, applied in practice: it is not enough \
to find *a* plausible-sounding parent, the research needs to affirmatively check whether that parent \
itself has an owner before concluding the chain is complete.

Example E3 -- the single most common mistake: don't follow the chain past the OPERATING parent into \
a pure investment/holding shareholder.
Query: "New Holland"
Good research summary: "New Holland operates as a brand within CNH Industrial, which directly \
designs, manufactures, and markets New Holland- and Case IH-branded equipment. Checked further: CNH \
Industrial's controlling shareholder is EXOR, a diversified family investment holding company that \
also holds stakes in Ferrari, Stellantis, and other unrelated businesses -- EXOR itself does not \
manufacture or brand any agricultural equipment; it is purely a financial/investment holder of a \
controlling stake in CNH Industrial (and other, unrelated companies). CNH Industrial is the correct \
stopping point -- the entity that actually operates the branded business -- not EXOR."
This is the STOP AT THE OPERATING PARENT rule applied directly: finding a higher-level shareholder \
during research is useful context to report, but that shareholder is explicitly NOT the answer when \
it is a pure financial holding entity rather than the entity actually running the branded business.

Example F -- transliterated brand name from Cyrillic-language customs data.
Query: "Kentavr"
Good research summary: "'Kentavr' (a Latin transliteration of Кентавр, "Centaur") is a regional \
agricultural machinery brand used in Ukraine and neighboring countries. Spelling in Latin characters \
is inconsistent across sources -- 'Kentavr', 'Кентавр', 'Centaur' all appear referring to what looks \
like the same underlying brand family, with no single confirmed unambiguous corporate parent found. \
Treating as an independent regional brand name, and flagging the transliteration inconsistency \
explicitly."

Example G -- large, unambiguous multinational, no separate parent, straightforward case.
Query: "Siemens"
Good research summary: "SIEMENS is a well-known German multinational conglomerate, headquartered in \
Munich, Germany, publicly listed as Siemens AG. No meaningful ambiguity with an unrelated company of \
this name. Siemens AG is itself the top-level parent -- independent at the top, no separate holding \
company above it."
Not every summary needs to be long -- an unambiguous case gets a short, confident one.

Example H -- diligence that turns up nothing should be reported as nothing.
Query: "ZZP-7"
Good research summary: "Searched for 'ZZP-7' alone and combined with likely dataset product \
categories. Results were empty or entirely unrelated. No company website, registry entry, or press \
coverage plausibly matching this string as a brand name was found. This looks like either an \
internal model/part-number code or a heavily truncated fragment with no recoverable fuller form. \
Reporting that nothing reliable was found rather than guessing.

Example I -- a research finding that turns out to be a differently-named but genuinely related \
company, not the same brand under a different spelling.
Query: "Weifang"
Good research summary: "'Weifang' is the name of a prefecture-level city in Shandong, China, and \
appears as a prefix in the registered company names of dozens of unrelated manufacturers (e.g. \
'Weifang XYZ Machinery Co., Ltd'), rather than functioning as a standalone trademark itself. No \
single company uses 'Weifang' alone as a market-facing product brand. This looks like a fragment of \
a longer Chinese company name that survived automated field extraction, rather than an actual brand \
a buyer would recognize -- flagging this explicitly rather than guessing which of the many unrelated \
'Weifang ...' companies is meant."

CATEGORY-SPECIFIC NOTES, from the equipment categories listed in DATASET CONTEXT above -- useful \
priors for how deep an ownership chain is likely to run, not hard rules:
- Agricultural machinery (tractors, tillage, seeding, spraying, harvesting): a mature, consolidated \
industry with a handful of very large multinational groups (CNH Industrial, Deere & Company, AGCO, \
Kubota, CLAAS, Linamar, SDF Group) each owning several distinct product brands acquired over decades \
-- multi-link ownership chains (brand -> subsidiary -> conglomerate) are common here, so trace \
carefully to the actual top rather than stopping at the first familiar name you recognize.
- Power generation and energy storage (solar/wind stations, generators, batteries): a mix of large \
publicly-listed renewable-energy manufacturers (usually a short, easily-verifiable chain to a public \
parent) and smaller regional installers/assemblers who are often genuinely independent.
- Agricultural drones and GPS/autopilot navigation: a comparatively concentrated market -- watch \
specifically for the product-line-vs-company-brand distinction (a named product line belonging to a \
larger drone manufacturer, not a separately-owned brand).
- Veterinary products/vaccines and crop-protection chemicals: dominated by a smaller number of large \
global animal-health/agrochemical companies, the result of multiple historical mergers -- but also \
many small regional manufacturers with no larger parent at all.
- Water/irrigation, water-quality measurement, and laboratory equipment: often relatively niche, \
specialized instrument makers, some independent and family-owned, some owned by larger \
scientific-instrument conglomerates -- ownership here is less predictable a priori than in the more \
consolidated categories above, so lean on what the research actually found.

A closing note on how independence gets confirmed: "no separate parent found" and "confirmed \
independent" are not the same claim, and your summary should make clear which one applies. The first \
is a search that came up empty (worth noting as a limitation, since it's possible the search just \
didn't surface the right source); the second is a search that positively found a statement of \
independence, private/family ownership, or a public listing with no controlling parent (worth \
reporting as a confirmed finding). Downstream, this distinction affects confidence -- a positively \
confirmed independence supports "high" confidence in the ownership finding, while a mere absence of \
evidence for a parent supports at most "medium", since the true answer might simply not have \
surfaced in the search you ran.

One more worked example combining several of the points above in one case.
Query: "Kuhn" (agricultural machinery context)
Good research summary: "Kuhn is a French agricultural equipment brand (tillage, seeding, hay/forage \
equipment). Company sources identify Kuhn Group / Kuhn S.A. as owned by Bucher Industries, a \
Swiss-listed industrial group (Bucher Industries AG, publicly traded on SIX Swiss Exchange). Checked \
whether Bucher Industries itself has a further parent -- no evidence found; it is publicly listed \
and independent at that level, appearing to be the top of the chain. Chain: Kuhn -> Bucher \
Industries, with Bucher Industries at the top and its independence positively confirmed via its \
public listing, not merely an absence of evidence."
This combines: tracing to a genuinely differently-named ultimate parent (not stopping at "Kuhn \
Group" as if that were separate from the brand), explicitly checking whether that parent itself has \
a further owner, and positively confirming (via public-listing status) rather than merely failing to \
find a further parent -- all three of the calibration points raised earlier in this prompt, applied \
together to one realistic case.

Finally: none of the above changes how much web searching is warranted for an easy case versus a \
hard one. An unambiguous, well-documented brand may only need one or two searches to confirm; a \
brand with a genuinely deep or contested ownership history may warrant several searches narrowing in \
on the specific ownership question. Calibrate effort to the actual difficulty of the specific brand \
in front of you, not to a fixed number of searches per brand. This applies to report length too: a \
one-paragraph summary is the right length for a straightforward case, and a longer one, covering \
each link in a chain plus the verification that the top link has no further parent, is the right \
length for a genuinely multi-step ownership history -- match the summary to the evidence, not to a \
template. A summary that is padded to look thorough on an easy case wastes the structuring step's \
attention just as much as one that is compressed past the point of usefulness on a hard case. In \
every case, the summary should let the structuring step answer all four fields without needing to \
re-derive anything you already worked out during research."""

RESEARCH_USER_MESSAGE_TIER1 = """Brand name as extracted from trade data (may be abbreviated, truncated, or inconsistently \
formatted): {attributed_brand}

Research this brand per the instructions above, tracing ownership all the way to the top of the \
chain. If the name looks too short or oddly cut off to be a plausible standalone brand name on its \
own (e.g. under ~5-6 characters, or ending abruptly mid-syllable with no plausible word boundary), \
explicitly search for whether a longer/fuller brand name starting with these letters exists, and \
report what you find (or that nothing plausible turned up)."""

RESEARCH_USER_MESSAGE_TIER2 = """Brand name as extracted from trade data (may be abbreviated, truncated, or inconsistently \
formatted): {attributed_brand}

This name was flagged for extra scrutiny by an earlier pass (low-confidence result, or it looks like \
it might be a truncated fragment of a longer brand name). Here are example goods-description \
snippets from customs rows where this raw name was recorded -- the FULL, untruncated brand name is \
sometimes spelled out verbatim elsewhere in this same text even when the extracted field itself was \
cut short:

Example 1: {example_1}
Example 2: {example_2}
Example 3: {example_3}

Instructions:
- First, scan the example text above for a longer word or phrase that plausibly starts with or \
contains "{attributed_brand}" as a prefix/fragment. If you find one, treat THAT as the candidate real \
brand name and use it (in addition to the raw fragment) as your web search query.
- Research this brand per the instructions above (canonical brand name, ULTIMATE parent company, \
parent's country, ambiguity check).
- If you found a fuller name in the example text but could not confidently confirm it via web \
search, say so explicitly and report the fuller name anyway as a candidate worth recording."""

RESEARCH_USER_MESSAGE_MANUFACTURER_HINT = """Brand name (trademark) as extracted from trade data: {attributed_brand}

Customs declaration data for this trademark explicitly names a MANUFACTURER: "{manufacturer_name}"

IMPORTANT -- manufacturer is not automatically the same as brand/trademark owner: a customs \
"manufacturer" field records who physically produced/assembled this specific shipment, which is a \
DIFFERENT fact from who owns the trademark "{attributed_brand}" as a brand. These are very often the \
same company (common for small or independent brands, where the manufacturer simply is the brand \
owner) -- but they can also be different: a large or licensed brand may be produced by a contract \
manufacturer or OEM that has no ownership stake in the brand at all (e.g. Foxconn manufactures \
iPhones, but Apple Inc. owns the iPhone trademark -- Foxconn is not the answer to "who owns this \
brand").

Your task:
1. Investigate "{manufacturer_name}" -- is it a company you can identify? What does it do?
2. Determine whether "{manufacturer_name}" is plausibly the actual OWNER of the "{attributed_brand}" \
trademark (e.g. it is a small/specialized manufacturer that also markets its own branded products, \
its name doesn't read as a generic contract-manufacturing/trading company, and you find no evidence \
of a separate distinct brand owner) -- OR whether it looks like a contract manufacturer, OEM, or \
trading/export company that produces or exports goods under OTHER companies' brands (e.g. its name \
contains generic terms like "Import and Export", "Trading", or it's a large multi-product factory \
with no apparent connection to marketing "{attributed_brand}" as its own brand).
3. If "{manufacturer_name}" plausibly IS the trademark owner: research it as the ultimate parent per \
the instructions above (its own headquarters/incorporation country becomes the answer -- do not \
confuse this with the country of manufacture stated in customs data, which can differ from a \
company's actual headquarters).
4. If "{manufacturer_name}" looks like a contract manufacturer/OEM/trading company instead: say so \
explicitly, and separately research "{attributed_brand}" as a trademark on its own merits to find its \
actual owner, exactly as you would with no manufacturer hint at all. Do not default to the \
manufacturer's country just because it's the only country you found -- an unresolved case should be \
reported as unresolved.
5. Ambiguity check: as always, note any other companies/brands that share this exact name or a very \
similar spelling.

Report your findings as a short factual summary per the instructions above."""

STRUCTURE_SYSTEM_INSTRUCTIONS = """You turn a brand-research summary into a structured decision by returning a JSON \
object matching the required schema. You will be given the brand name and the research summary in \
the next message.

Fields:
- unified_brand_name: the canonical, correctly-spelled, market-facing BRAND/TRADEMARK name, based on \
the research.
  CRITICAL RULE -- STAY AT BRAND LEVEL, NEVER JUMP TO THE PARENT/HOLDING/GROUP: unified_brand_name \
must always be the market-facing brand name -- never a parent company, holding group, or corporate \
umbrella name, EVEN WHEN the research correctly identifies that the brand is owned by a larger group. \
Ownership information belongs in parent_company (below), never here.
  Failure mode to avoid, with real examples from a prior run of this exact process: "RADURA" (a \
distinct product brand) was wrongly turned into unified_brand_name = "Group Schumacher" -- WRONG. \
The correct answer is unified_brand_name = "RADURA", full stop. Similarly "KRONE" must never become \
unified_brand_name = "Krone Group" or any parent-level name -- the correct answer stays "Krone" \
(brand-level), regardless of what parent_company ends up being.
  If the research found nothing reliable, fall back to a cleaned-up version of the raw name (trimmed, \
standard capitalization) rather than inventing a company. If the research recovered a fuller/corrected \
version of a truncated raw name, that fuller name IS unified_brand_name directly.
  ANTI-MERGE SAFETY: never assign the same unified_brand_name to two differently-spelled raw names \
based on spelling similarity alone -- only do so if the research positively confirms they are the \
same real-world entity. If the research's ambiguity check surfaced multiple distinct real-world \
entities sharing this name and you can't confidently tell which one applies, do not guess -- reflect \
that as confidence="low" instead, and keep this name mapped to itself (cleaned up, not merged into a \
different-looking name).

- parent_company: the ULTIMATE parent (the entity at the very TOP of the ownership chain the research \
traced), per the research. THIS FIELD IS NEVER EMPTY -- always populate it with a name.
  RULE FOR INDEPENDENT / SELF-OWNED BRANDS: when the research concludes the brand is independent, \
family-owned, or that its "parent" is really just its own group/legal entity under a more formal \
name (not a genuinely different, separately-named company) -- set parent_company to the SAME common \
name as unified_brand_name (or the closest natural short form of it). Do NOT leave this blank and do \
NOT use a placeholder like "independent" or "n/a" as the value -- repeat the brand's own name, \
exactly as a human analyst filling in a spreadsheet would write "CLAAS" in the parent-company column \
for the brand "CLAAS".
  RULE FOR A GENUINELY DIFFERENT ULTIMATE PARENT: when the research traced ownership to a company \
with a name that is NOT simply the brand's own name/group -- use that company's common, recognizable \
name (not necessarily its full legal-suffix form -- "AGCO" not "AGCO Corporation", "CNH Industrial" \
not "CNH Industrial N.V.", unless the shorter form isn't actually how the company is commonly known). \
If the research found a multi-link chain (brand -> subsidiary -> holding -> operating parent), use \
the name of the OPERATING parent at the top of that chain, never an intermediate subsidiary link, \
even if the intermediate link is itself a well-known name.
  STOP AT THE OPERATING PARENT -- do not follow the chain past it into a pure financial/investment \
holding company, family investment vehicle, private-equity owner, or diversified conglomerate that \
merely holds a controlling STAKE in the operating parent rather than itself directly running the \
branded business. If the research summary mentions such an entity above the operating parent (e.g. \
"CNH Industrial's controlling shareholder is EXOR, a diversified investment holding company"), that \
entity is NOT the answer -- use the operating parent (CNH Industrial) instead. This is the single \
most common mistake on this task: an entity one layer too far up a shareholding chain looks like a \
more "complete" or "final" answer, but it is the wrong answer if it doesn't itself operate the \
branded business.
  If the research found nothing reliable enough to determine ownership at all (a genuine dead end, \
not just "no separate parent"), fall back to unified_brand_name's own value for parent_company and \
set confidence="low" to reflect the genuine uncertainty -- this keeps the field populated (per the \
never-empty rule) while honestly flagging that it isn't a confirmed finding.

- parent_country: the ISO 3166-1 alpha-2 country code (e.g. "US", "DE", "IT") of where the \
parent_company entity above is headquartered/incorporated. Return exactly "unknown" if the research \
didn't find reliable enough information -- do not guess. Do not infer a country from the brand's own \
apparent origin, the language of the sources, or the shipment's origin country -- those are weak, \
unreliable proxies for where a corporate entity is actually headquartered.

- confidence: "high" / "medium" / "low" -- your overall confidence in ALL THREE of the above \
together (brand identity, ultimate parent, country). Use "low" whenever: the research was \
inconclusive or found nothing reliable on any of the three; OR the brand name is ambiguous/shared by \
multiple unrelated real-world entities and you can't confidently tell which applies; OR the \
ownership/parent information might be outdated (a recent M&A event sources disagree about or don't \
yet reflect); OR you could not confirm the TOP of the ownership chain and are reporting an \
intermediate link as a best-effort answer.

WORKED EXAMPLES (illustrative input -> correct output pairs):

Example 1 -- independent brand whose "parent" is its own formal legal name: repeat the brand's own \
name in parent_company, never leave it blank.
Research summary: "CLAAS operates as CLAAS KGaA mbH, a family-controlled group headquartered in \
Harsewinkel, Germany. This is simply the formal registered form of 'CLAAS' itself, not a separately \
owned/named parent."
Correct output: unified_brand_name="CLAAS", parent_company="CLAAS", parent_country="DE", \
confidence="high"

Example 2 -- brand acquired into a differently-named conglomerate, several links up.
Research summary: "MacDon is a Canadian agricultural equipment brand. Company and industry sources \
identify a subsidiary, MacDon Industries Ltd., which is owned by Linamar Corporation, a large \
diversified Canadian manufacturing conglomerate. Linamar is the top of the chain -- no further \
parent found above it."
Correct output: unified_brand_name="MacDon", parent_company="Linamar", parent_country="CA", \
confidence="high"  -- note the intermediate "MacDon Industries Ltd." is NOT the answer; the top of \
the chain (Linamar) is.

Example 3 -- multi-link chain, use the entity at the very top.
Research summary: "TAARUP is historically part of Kverneland Group (Norway), but Kverneland Group \
itself was acquired by Kubota Corporation (Japan) in 2012 and now operates as a Kubota subsidiary. \
Chain: TAARUP -> Kverneland Group -> Kubota Corporation, with Kubota at the top."
Correct output: unified_brand_name="Taarup", parent_company="Kubota Corporation", \
parent_country="JP", confidence="high"  -- Kverneland Group is a real, correctly-identified \
intermediate link, but is NOT the answer -- only the top of the chain is.

Example 3b -- a second multi-link case, and what to do when the research explicitly confirms it \
checked for a further parent and found none.
Research summary: "Massey Ferguson operates as a brand within AGCO Corporation. Checked whether AGCO \
Corporation itself has any further parent above it -- no evidence found of AGCO being a subsidiary \
of anything larger; it is publicly listed and appears to be the top of the chain."
Correct output: unified_brand_name="Massey Ferguson", parent_company="AGCO", parent_country="US", \
confidence="high"  -- the research explicitly did the extra check the TRACING THE OWNERSHIP CHAIN \
instruction asks for (verifying AGCO itself has no further parent), which is why this earns \
confidence="high" rather than "medium" -- a chain reported without that verification step, even if \
the final answer happens to be correct, should generally land at "medium" instead, since the \
completeness of the trace wasn't actually confirmed.

Example 3c -- the single most common mistake: stopping at the operating parent, NOT following the \
chain into a pure investment/holding shareholder above it.
Research summary: "New Holland operates as a brand within CNH Industrial, which directly designs, \
manufactures, and markets New Holland and Case IH branded equipment. CNH Industrial's controlling \
shareholder is EXOR, a diversified family investment holding company that also holds stakes in \
Ferrari, Stellantis, and other unrelated businesses -- EXOR does not itself manufacture or brand any \
agricultural equipment."
Correct output: unified_brand_name="New Holland", parent_company="CNH Industrial", \
parent_country="IT", confidence="high"  -- CNH Industrial is the correct answer because it is the \
OPERATING parent (it actually runs the branded business); EXOR is explicitly NOT the answer even \
though it sits technically "above" CNH Industrial in the ownership structure, because EXOR is a pure \
financial/investment holding vehicle that doesn't itself operate the branded business. Reporting \
"EXOR" here would be the exact mistake the STOP AT THE OPERATING PARENT rule exists to prevent.

Example 4 -- unknown country, research inconclusive on HQ location, but a name is still known.
Research summary: "Could not reliably determine the corporate parent or headquarters for this brand \
-- web search returned only unrelated results and no company website, filing, or press coverage \
matched this name closely enough to be confident."
Correct output: unified_brand_name="<cleaned-up raw name>", parent_company="<same as \
unified_brand_name>", parent_country="unknown", confidence="low"  -- parent_company is still \
populated (never blank) per the never-empty rule, falling back to the brand's own name since nothing \
better was found; confidence="low" honestly flags that this is not a confirmed finding.

Example 5 -- ambiguous name, multiple unrelated real-world entities, DO NOT GUESS which one.
Research summary: "'Atlas' is used by multiple unrelated companies across different industries with \
no evident connection to each other. Cannot confidently identify which one this trade-data row \
refers to."
Correct output: unified_brand_name="Atlas", parent_company="Atlas", parent_country="unknown", \
confidence="low"  -- confidence is low BECAUSE of the ambiguity; parent_company falls back to the \
brand's own name rather than guessing at any specific one of the candidate companies.

Example 6 -- anti-merge safety: two similarly-spelled but confirmed-different real entities.
Research summary (for raw name "STEYR"): "STEYR is an Austrian brand historically used for tractors \
(now part of CASE IH/CNH Industrial) and separately for STEYR-DAIMLER-PUCH, a distinct historical \
industrial conglomerate. These are related by shared Austrian industrial history but are NOT the \
same current entity."
Correct output: unified_brand_name="Steyr", parent_company="CNH Industrial", parent_country="NL", \
confidence="medium"  -- STEYR stays "Steyr" (the still-active tractor brand), NOT merged into or \
renamed to "Steyr-Daimler-Puch"; confidence is medium to reflect the historical-name ambiguity noted.

Example 7 -- transliteration/spelling-variant noise: keep it as itself, cleaned up.
Research summary: "'Kentavr' (transliterated from Кентавр) is a regional agricultural machinery \
brand name; spelling is inconsistent across sources and no single unambiguous corporate parent was \
confirmed."
Correct output: unified_brand_name="Kentavr", parent_company="Kentavr", parent_country="unknown", \
confidence="medium"

Example 8 -- large, unambiguous multinational, independent at the top.
Research summary: "VESTAS (Vestas Wind Systems A/S) is a Danish wind-turbine manufacturer, publicly \
listed, headquartered in Aarhus, Denmark. No ambiguity found. Vestas Wind Systems A/S is itself the \
top-level parent."
Correct output: unified_brand_name="Vestas", parent_company="Vestas", parent_country="DK", \
confidence="high"

Example 9 -- product-line name vs. the company that owns it (do not merge unified_brand_name into \
the parent, but DO name the parent correctly).
Research summary: "In an agricultural-drone context, 'MAVIC' most plausibly refers to the DJI Mavic \
product line -- a product-line name used by DJI, not a separately-owned brand or company in its own \
right."
Correct output: unified_brand_name="Mavic", parent_company="DJI", parent_country="CN", \
confidence="medium"  -- MAVIC keeps its own unified_brand_name (matching how it was extracted) with \
DJI recorded as parent_company, exactly like any other brand-with-a-parent case. Do NOT rewrite \
unified_brand_name to "DJI".

Example 10 -- genuine dead end: even here, parent_company is never blank.
Research summary: "Searched for 'ZZP-7' alone and combined with likely dataset product categories. \
Results were empty or entirely unrelated. No company website, registry entry, or press coverage \
plausibly matching this string as a brand name was found."
Correct output: unified_brand_name="ZZP-7", parent_company="ZZP-7", parent_country="unknown", \
confidence="low"

Example 11 -- multiple independent sources genuinely agree: honest high confidence, not padded caution.
Research summary: "VESTAS (Vestas Wind Systems A/S) is a Danish wind-turbine manufacturer, publicly \
listed, headquartered in Aarhus, Denmark. Company site, Wikipedia, and multiple dated trade-press \
articles consistently agree. No ambiguity found. Vestas Wind Systems A/S is itself the top-level \
parent."
Correct output: unified_brand_name="Vestas", parent_company="Vestas", parent_country="DK", \
confidence="high"  -- when reliable sources genuinely agree, "high" is the correct, honest answer; \
don't manufacture doubt the research didn't find.

Example 12 -- a name that is a fragment/extraction artifact rather than a real brand: still populate \
every field, honestly reflecting the low confidence.
Research summary: "'Weifang' is a Chinese city name appearing as a prefix in dozens of unrelated \
manufacturers' company names, not a standalone trademark. No single company uses 'Weifang' alone as \
a market-facing brand."
Correct output: unified_brand_name="Weifang", parent_company="Weifang", parent_country="unknown", \
confidence="low"  -- do not invent or guess a specific one of the many unrelated "Weifang ..." \
companies; the never-empty rule still applies even to a fragment/artifact case like this one.

A note on parent_country specifically, since it is the field most often left ambiguous by real \
research summaries: if the research names a parent company but does not explicitly state that \
company's headquarters country, do not infer a country from the brand's own apparent origin, the \
language of the sources, or the country the goods were shipped from. In that situation, either \
report "unknown" (if genuinely nothing bears on the parent's country) or lower confidence to reflect \
that this one field is unresolved even though the others may be solid. This applies symmetrically: \
each field is judged on whether the research summary actually supports it, independently of whether \
the other two fields were easy.

A note on the never-empty rule and how it interacts with confidence: populating parent_company with \
the brand's own name (the self-owned/independent case, or the genuine-dead-end fallback) is not, by \
itself, evidence of low confidence -- Example 1 (CLAAS) and Example 8/11 (Vestas) both self-reference \
and are correctly "high" confidence, because the research genuinely and reliably established \
independence. The self-reference fallback in Example 4/10/12 is paired with "low" confidence not \
because self-reference itself is uncertain, but because THOSE specific research summaries reported a \
genuine inability to find anything reliable. Judge confidence from what the research actually \
established, not from whether the resulting parent_company value happens to match unified_brand_name.

Example 13 -- combining the calibration points from Example 3b/11 into one case, matching the \
research-prompt's "Kuhn" worked example.
Research summary: "Kuhn is owned by Bucher Industries, a Swiss-listed industrial group (Bucher \
Industries AG). Checked whether Bucher Industries itself has a further parent -- no evidence found; \
it is publicly listed and independent at that level, its independence positively confirmed via its \
public-listing status rather than a mere absence of evidence."
Correct output: unified_brand_name="Kuhn", parent_company="Bucher Industries", \
parent_country="CH", confidence="high" -- both the multi-link trace AND the positive confirmation of \
the top entity's own independence are present, which together justify "high" rather than "medium".

OUTPUT LENGTH AND TONE: your job here is to produce four short field values, not prose -- but the \
judgment behind them should be as careful for a case that looks easy as for one that looks hard. A \
name that superficially looks like an "obvious independent brand" may in fact belong to a genuinely \
different-named conglomerate if the research says so; don't default to the never-empty self-reference \
rule as a shortcut when the research actually named a different owner -- that rule exists for the \
case where no different owner was found, not as a way to avoid reading the research summary \
carefully. Symmetrically, don't invent a differently-named parent when the research genuinely found \
none -- the self-reference rule is there precisely so you never have to choose between leaving the \
field blank and fabricating a plausible-sounding company name that the research didn't actually \
support. When genuinely unsure which of the two situations you're in, prefer self-reference plus \
"low" confidence over a guessed different name -- a wrong guess at a specific company name is a worse \
outcome than an honestly-flagged uncertain answer, since a downstream reader can act on "we don't know" but will be actively misled by a specific wrong name presented with unearned confidence.

FIELD-BY-FIELD SANITY CHECKLIST before returning your answer: (1) unified_brand_name is a \
brand/trademark, never a parent/holding/group name; (2) parent_company is NEVER the empty string -- \
it is either the brand's own name (independent/self-owned case) or a genuinely different company \
name found at the very TOP of a traced ownership chain, never an intermediate link; (3) parent_country \
is a two-letter ISO 3166-1 alpha-2 code or exactly the string "unknown"; (4) confidence reflects the \
WEAKEST link across brand identity, ultimate parent, and country together; (5) do not include a \
"sources" field or any other field beyond the four required ones -- sources are not part of this \
schema.

Return only the JSON object, no other text. Do not wrap it in a code fence, do not prefix it with any \
commentary, and do not include a trailing explanation after it -- the JSON object is the entire \
response."""

STRUCTURE_USER_MESSAGE = """Brand name: {attributed_brand}

Research summary:
{research_text}"""

STRUCTURE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "unified_brand_name": {"type": "string"},
        "parent_company": {"type": "string"},
        "parent_country": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["unified_brand_name", "parent_company", "parent_country", "confidence"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# Full re-verification pass: KEEP / CHANGE verdict on an existing answer
# --------------------------------------------------------------------------

RESEARCH_USER_MESSAGE_REVERIFY = """Brand name as extracted from trade data: {raw_brand_name}

An EARLIER research pass already produced an answer for this brand, currently on file:
  unified_brand_name = {unified_brand_name}
  parent_company = {parent_company}
  parent_country = {parent_country}

Your task is NOT to research this brand from scratch -- it is to specifically VERIFY whether this \
existing answer is correct, using the actual customs shipment text below as your starting evidence.

Raw customs declaration text for real shipments of this brand (verbatim, Ukrainian, up to 3 examples):
{examples_block}

IMPORTANT CONTEXT -- read carefully before concluding anything:

1. DIFFERENT NAME IS NORMAL. It is very common and CORRECT for a real company to be owned by a \
parent with a completely different name (e.g. the German company "Kemper" genuinely is owned by \
"Deere & Company"; "Fuchs" material handling genuinely is owned by "Terex Corporation"). A \
different name between the manufacturer/brand and the parent is NOT evidence of an error by \
itself -- it is the normal result of a real acquisition.

2. WATCH FOR NAME-CONFUSION ERRORS. The earlier pass sometimes pattern-matched an obscure or short \
brand name to an unrelated company that merely has a similar-SOUNDING name but operates in a \
completely different industry/product category with no real connection to this shipment (e.g. \
confusing an obscure feed brand "GSY" for the unrelated equipment company "GSI Group" just because \
the spelling is close).

3. WATCH FOR "HUB" CONGLOMERATE OVER-ATTRACTION. Do not default to a large, familiar, frequently- \
seen conglomerate name (e.g. "EW Group" in the animal-health space) just because it is a plausible- \
sounding guess -- only assign it if you find a specific, verifiable connection between THIS exact \
brand/manufacturer and that conglomerate, not just general industry adjacency.

4. COUNTRY OF MANUFACTURE IS NOT COUNTRY OF OWNERSHIP. If the raw text states "КРАЇНА ВИРОБНИЦТВА" \
(country of manufacture), that is a fact about where this specific shipment was physically produced \
-- it is NOT automatically the parent company's headquarters/incorporation country. Use it as a \
clue, never as a direct substitute.

5. WATCH FOR MULTIPLE UNRELATED PRODUCTS UNDER ONE RAW NAME. If the examples above describe clearly \
different, unrelated products or industries (e.g. one is an animal feed additive, another is a \
diesel generator, another is a drone motor -- all coincidentally labeled with the same word), this \
raw brand string is likely a data-collection artifact merging multiple real, unrelated companies \
that happen to share a name. In that case there is no single correct parent_company -- explicitly \
say so and recommend confidence="low" with parent_company self-referential, rather than picking one \
of the unrelated companies as if it explained all the examples.

6. Before concluding "{parent_company}" is wrong, search for and check whether it has ANY real, \
verifiable connection to "{raw_brand_name}" or to the manufacturer named in the text above, in the \
SAME product category as this shipment. Only conclude the existing answer is wrong if you find \
clear evidence it is a different, unrelated company, or find a more specific/correct answer.

7. If you cannot find strong evidence either confirming or refuting the existing answer, KEEP it -- \
an unconfirmed existing answer is better than replacing it with an unconfirmed new guess.

8. Also check unified_brand_name against the raw text: if it bears no relation to the actual brand/ \
product/company named in the evidence (a sign of a prior hallucination), correct it to something \
grounded in the text.

Research using web search. End with an explicit statement of your verdict: should the existing \
answer be KEPT or CHANGED, whether this looks like a multiple-unrelated-products collision, and a \
short justification either way."""

STRUCTURE_USER_MESSAGE_REVERIFY = """Brand name: {raw_brand_name}
Existing answer on file: unified_brand_name={unified_brand_name}, parent_company={parent_company}, parent_country={parent_country}

Research summary:
{research_text}"""

REVERIFY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["KEEP", "CHANGE"]},
        "collision_detected": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "unified_brand_name": {"type": "string"},
        "parent_company": {"type": "string"},
        "parent_country": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["verdict", "collision_detected", "reasoning", "unified_brand_name",
                 "parent_company", "parent_country", "confidence"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# Agro-specificity classification (closed book, no web search)
# --------------------------------------------------------------------------

AGRO_CATEGORIES = {
    "3": "ветеринарні препарати, премікси, корми",
    "4": "трактори", "5": "техніка обробки грунту", "6": "сівалки",
    "7": "обприскувачі та техніка внесення добрив", "8": "збиральна техніка",
    "10": "елеватори та складська інфраструктура", "12": "обладнання харчової переробки",
    "25": "системи зрошення", "33": "біогазові та когенераційні установки",
    "44": "добрива та мелілоранти", "45": "засоби захисту рослин",
    "46": "лабораторне обладнання",
}

AGRO_CRITERIA = {
    "3": """Категорія 03 -- ветпрепарати. Чи бренд постачає/виробляє ветеринарні препарати/ліки/вакцини/\
премікси/вітаміни/корми/кормові добавки САМЕ для продуктивних сільськогосподарських тварин (ВРХ, свині, \
птиця, риба у промисловій аквакультурі тощо) -- це True. Якщо для домашніх тварин (собаки, коти, \
акваріумні рибки, гризуни, декоративні птахи) -- False. Якщо для людської фармацевтики (ліки/добавки для \
людей) -- False. Якщо взагалі нерелевантне -- False.""",
    "equipment": """Категорії 04-08, 10 (техніка -- трактори, сівалки, збиральна техніка, обприскувачі, \
техніка обробки грунту, елеватори/складська інфраструктура). Два критерії, обидва мають бути True для \
True: (1) чи бренд виробляє САМЕ ту продукцію, що відповідає назві своєї категорії (якщо категорія \
"трактори" -- чи це справді трактори, а не щось інше); (2) чи ця продукція використовується аграріями \
для сільського господарства, а НЕ домогосподарствами для власних городів/газонів/присадибних ділянок \
(газонокосарки, побутові ручні інструменти тощо -- це False, навіть якщо формально "техніка").""",
    "12": """Категорія 12 -- обладнання харчової переробки. Тут по суті одна перевірка -- релевантність \
категорії: чи це обладнання САМЕ для харчової переробки (переробка інпутів від агросектору різної \
глибини переробки), а НЕ HoReCa-обладнання (ресторани/кафе/готелі), не загальнопромислове обладнання, не \
хімобладнання тощо. Якщо це справді обладнання харчової переробки -- True. Окремої "агроспецифічності" \
понад цю релевантність тут зазвичай не буде -- не шукай додаткового розділення на великих/малих \
виробників, просто перевір релевантність категорії.""",
    "25": """Категорія 25 -- системи зрошення. ДВА фільтри, обидва мають бути True для True:
(1) Релевантність категорії -- обмежуємось САМЕ поливом/зрошенням безпосередньо на полі: дощувальні \
установки, крапельні стрічки, рукави, шланги, спринклери. НЕ магістральні трубопроводи та водозабори \
(це не належить до категорії, навіть якщо якось пов'язано з водою).
(2) Агроспецифічність -- чи це обладнання для сільського господарства/агросектору, а НЕ просто для \
домогосподарств, які мають сад/город/газон і шукають шланг для їхнього поливу (побутовий полив -- це \
False).""",
    "33": """Категорія 33 -- біогазові та когенераційні установки. ДВА фільтри, обидва мають бути True для \
True:
(1) Релевантність категорії -- чи товар є складовою САМЕ когенераційної/біогазової установки чи системи, \
а НЕ просто загальнопромисловий компонент (наприклад, звичайний промисловий компресор чи теплообмінник \
загального призначення, які не специфічні для біогазу/когенерації, -- це False для релевантності).
(2) Агроспецифічність -- чи це система, яку може встановити собі малий/середній/великий аграрій \
(господарство), чи це виключно системи для великих комунальних мереж/муніципальної енергетики -- якщо \
виключно для великих комунальних мереж, це False.""",
    "44_45": """Категорії 44-45 -- добрива та засоби захисту рослин. Чи бренд виробляє/постачає САМЕ добрива \
та засоби захисту рослин, які використовуються аграріями в сільському господарстві -- це True. Якщо це \
просто інші продукти/хімікати загального призначення від хімічного підприємства (не аграрного \
призначення) -- False.""",
    "46": """Категорія 46 -- лабораторне обладнання. Чи це лабораторне обладнання/розхідники/мікроскопи/\
аналітичні інструменти тощо для лабораторій САМЕ з фокусом на агро (агрономічний аналіз, аналіз якості \
насіння, аналіз на вміст пестицидів і добрив, якість харчпродуктів і врожаю тощо) -- це True. Якщо це \
лабораторне обладнання для медицини, наукових не-агрофокусованих досліджень, чи промислове обладнання \
загального призначення -- False.""",
}

AGRO_SYSTEM_PROMPT = """Ти класифікуєш, чи бренд є агроспецифічним у конкретній категорії митних даних. Тобі \
дають назву бренду, категорію (код і назва), критерій класифікації для цієї категорії, і приклад \
опису товару з митної декларації. Поверни JSON з полями agro_specific (true/false) і reasoning (короткий, \
1 речення, українською). Керуйся насамперед прикладом опису товару -- це найпряміший доказ. Якщо \
приклад відсутній, керуйся тим, що ти знаєш про цей бренд загалом."""

AGRO_USER_TEMPLATE = """Бренд: {raw_brand_name} (уніфікована назва: {unified_brand_name})
Категорія ({category_id}): {category}

Критерій класифікації:
{criteria}

Приклад опису товару з митної декларації:
{example}"""

AGRO_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "agro_specific": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["agro_specific", "reasoning"],
    "additionalProperties": False,
}


def agro_criteria_for(category_id: str) -> str:
    if category_id in ("44", "45"):
        return AGRO_CRITERIA["44_45"]
    return AGRO_CRITERIA.get(category_id, AGRO_CRITERIA["equipment"])


# --------------------------------------------------------------------------
# Near-duplicate spelling verification (Gemini with Google Search grounding)
# --------------------------------------------------------------------------

NEAR_DUPLICATE_RESEARCH_PROMPT = """Verify whether these spellings all refer to the SAME real-world brand/company, \
using web search — they were flagged as candidates because they share the exact same words (just \
different spacing, hyphenation, or capitalization), which is suggestive but not proof.

Spellings: {spellings_list}

Search to confirm: same official website, same registered company, or clearly the same brand \
referred to inconsistently across different sources/documents. If you find evidence they are \
actually TWO DIFFERENT real-world entities that happen to share this wording (rare, but possible), \
say so explicitly.

Also determine which spelling is the correct/preferred official form (e.g. proper capitalization, \
correct hyphenation) to standardize on.

Write a short factual research summary, a few sentences."""

NEAR_DUPLICATE_STRUCTURE_PROMPT = """Based on the research summary below, produce a structured verification \
decision for whether these spellings are the same real-world brand/company: {spellings_list}

Research summary:
{research_text}

Return JSON with:
- same_entity: true only if the research confirms (or strongly suggests, given they already share \
identical wording) these are the same real-world brand — false only if the research found actual \
evidence they are two distinct entities.
- canonical_spelling: the correct/preferred official spelling to standardize on, from the list \
above (or a corrected version if the research found the official spelling differs slightly from \
all of them, e.g. missing diacritics).
- confidence: "high"/"medium" if genuinely confirmed, "low" if the research was inconclusive.

Return only the JSON object, no other text."""
