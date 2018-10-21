--srednia kwota rek oryginalnej na osobę we wnioskach odrzuconych
SELECT count(1) as licza_odrzuconych,
  avg(kwota_rekompensaty_oryginalna/liczba_pasazerow)
    as_srednia_odrzuconych_na_os
FROM wnioski
WHERE stan_wniosku like 'odrz%';


--miary kwota rek oryginalnej na osobę we wnioskach odrzuconych
SELECT
  avg(kwota_rekompensaty_oryginalna/liczba_pasazerow) as srednia_rek,
  stddev(kwota_rekompensaty_oryginalna/liczba_pasazerow) as odch_rek,
  percentile_disc(0.25) WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow) as q1,
  percentile_disc(0.5) WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow) as q2,
  percentile_disc(0.75) WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow) as q3,
  mode()  WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow) as dominanta
FROM wnioski
WHERE stan_wniosku LIKE 'odrz%';

--miary kwota rek oryginalnej na osobę we wnioskach nieodrzuconych
SELECT
  avg(kwota_rekompensaty_oryginalna/liczba_pasazerow) as srednia_rek,
  stddev(kwota_rekompensaty_oryginalna/liczba_pasazerow) as odch_rek,
  percentile_disc(0.25) WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow) as q1,
  percentile_disc(0.5) WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow) as q2,
  percentile_disc(0.75) WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow) as q3,
  mode()  WITHIN GROUP (ORDER BY kwota_rekompensaty_oryginalna/liczba_pasazerow) as dominanta
FROM wnioski
WHERE stan_wniosku NOT LIKE 'odrz%';
--wniosek: brakistotnych różic w miarach dla wniosków odrzuconych i nie
