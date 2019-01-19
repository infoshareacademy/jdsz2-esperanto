-- percentyle kwoty rek. oryg. na pasażera dla wniosków odrzuconych

WITH pas_roz AS (
    SELECT
      id,
      kwota_rekompensaty_oryginalna / liczba_pasazerow
      - (kwota_rekompensaty / liczba_pasazerow) AS roznica
    FROM wnioski
    ORDER BY 2 DESC
)

SELECT percentile_disc(0.999)
    WITHIN GROUP (ORDER BY roznica)

    FROM
      pas_roz;

SELECT count(1)
FROM wnioski
WHERE kwota_rekompensaty_oryginalna / liczba_pasazerow
      - (kwota_rekompensaty / liczba_pasazerow) > 200;


SELECT unnest(
  percentile_disc(ARRAY [0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])
WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow))
as kwantyl_kwota_org_na_pas,
unnest(ARRAY [0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])
AS rzad_kwantylu
FROM wnioski;
