--procent odrzuconych wniosków dla poszczegółnych partnerów
-- wniosek: kiribati, tui i null odrzucają ok 10% wniosków,
-- pozostali poniżej procenta
SELECT partner, count(
    case WHEN stan_wniosku LIKE 'odrz%' then id end)/
  count(1)::NUMERIC AS procent_odrzuconych,
  count(id)
FROM wnioski
GROUP BY partner
ORDER BY 2 DESC ;


