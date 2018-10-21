-- średnia różnicy kwot rekompensaty dla >100
SELECT avg(kwota_rekompensaty_oryginalna-wnioski.kwota_rekompensaty)
as srednia_roznicy
  FROM wnioski
  WHERE kwota_rekompensaty_oryginalna - kwota_rekompensaty > 100;


-- procent wniosków, gdzie występuje różnica kwot
SELECT sum(
    CASE WHEN kwota_rekompensaty_oryginalna-kwota_rekompensaty > 0
      THEN 1
      ELSE 0
    END )/count(1)::NUMERIC as procent_z_roznica
FROM wnioski;

SELECT count(1) AS liczba_wnioskow_z_roznica
FROM wnioski
WHERE kwota_rekompensaty_oryginalna-wnioski.kwota_rekompensaty > 0;

--różnica w kwotach rekompensaty przeliczona na pasażera
SELECT id,
  (kwota_rekompensaty_oryginalna-wnioski.kwota_rekompensaty)
  /liczba_pasazerow as roznica_na_pasazera
  FROM wnioski
  ORDER BY roznica_na_pasazera DESC;

-- wniosek: na tyle mały procent wniosków z różnicami,
-- że nie warto się tym zajmować
