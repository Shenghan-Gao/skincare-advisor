# Ingredient Rule Sources

These rules are **cosmetic recommendation heuristics**, not a medical treatment protocol. They are used to ground retrieval/reward signals and to add safety warnings. Product formulation, concentration, route of use, allergies, and individual skin response matter; an ingredient name alone cannot establish clinical suitability.

## 1. Acne-related actives

- American Academy of Dermatology (AAD), **Acne clinical guideline**. The guideline supports topical benzoyl peroxide, topical retinoids, salicylic acid, and azelaic acid.  
  https://www.aad.org/member/clinical-quality/guidelines/acne
- AAD, **Acne: Diagnosis and treatment**. Discusses benzoyl peroxide and azelaic acid and notes that azelaic acid can also help post-acne dark spots.  
  https://www.aad.org/public/diseases/acne/derm-treat/treat

The broader `acne` mapping also includes cosmetic ingredients commonly used for oil/clogging/exfoliation support. Those entries should be interpreted as retrieval cues, not as claims that every product containing them treats acne.

## 2. Dark spots / uneven tone

- AAD, **Melasma: Diagnosis and treatment**. Discusses hydroquinone, tretinoin, azelaic acid, kojic acid, and vitamin C for pigment concerns.  
  https://www.aad.org/public/diseases/a-z/melasma-treatment
- AAD, **Dermatologist-recommended skin care for your 20s**. Notes science-backed roles for vitamin C and retinoids in dark spots and visible skin aging.  
  https://www.aad.org/public/everyday-care/skin-care-basics/care/skin-care-in-your-20s

## 3. Dryness / barrier support

- AAD, **Dermatologists' top tips for relieving dry skin**. Lists moisturizer ingredients including jojoba oil, dimethicone, glycerin, hyaluronic acid, lactic acid, lanolin, mineral oil, petrolatum, and shea butter.  
  https://www.aad.org/public/everyday-care/skin-care-basics/dry/dermatologists-tips-relieve-dry-skin
- AAD, **Isotretinoin: The truth about safety, side effects, and skin care**. Notes humectants such as hyaluronic acid and glycerin, ceramides, and squalane for hydration/barrier support.  
  https://www.aad.org/public/diseases/acne/derm-treat/isotretinoin/side-effects

## 4. Redness / sensitive-skin irritants

- AAD, **7 rosacea skin care tips dermatologists recommend**. Advises people with rosacea-prone sensitive skin to avoid alcohol, camphor, fragrance, glycolic acid, lactic acid, menthol, sodium lauryl sulfate, and urea when those ingredients irritate them.  
  https://www.aad.org/public/diseases/rosacea/triggers/tips

Because irritation depends on formulation and the individual, `common_irritants` produces warnings rather than automatic removal for all users.

## 5. Pregnancy safety

- AAD, **Dermatologist-approved pregnancy skin care** (2025). Advises avoiding retinoids and hydroquinone during pregnancy and using high-dose salicylic acid (>2%) sparingly / with clinician guidance.  
  https://www.aad.org/public/everyday-care/skin-care-secrets/routine/pregnancy-skin-care
- AAD, **Is any acne treatment safe to use during pregnancy?** Notes that experts generally recommend stopping adapalene and tretinoin; azelaic acid is thought to be safe; salicylic acid is generally considered safe for limited use; benzoyl peroxide is considered acceptable in limited amounts with clinician discussion.  
  https://www.aad.org/public/diseases/acne/derm-treat/pregnancy

For this reason, `pregnancy_unsafe` contains retinoid-family strings and hydroquinone. Salicylic acid and benzoyl peroxide are **not hard-blocked**; they are in `pregnancy_caution` so the app can warn the user instead of making an over-broad safety claim.

## 6. Pore appearance / wrinkles / other cosmetic mappings

These mappings are intentionally conservative retrieval heuristics. For example, salicylic acid and retinoids address clogging/texture, while moisturizers and retinoids can improve the appearance of photoaged skin. The system should say "may help the appearance of" rather than claiming to permanently shrink pores or medically treat disease.

## 7. Comedogenicity caveat

Ingredient-level "comedogenic" lists are imperfect because final formulation, concentration, vehicle, and individual response matter. The entries in `comedogenic` and `skin_type_to_avoid` are therefore **soft caution signals only**. They should never be presented as universal facts or used to diagnose acne.
