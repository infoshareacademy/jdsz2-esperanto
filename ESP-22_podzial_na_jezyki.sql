--procent odrzuconych w podziale na jezyki

SELECT jezyk,count(1), count(
   case WHEN stan_wniosku LIKE 'odrz%' then 1 end)/
 count(1)::NUMERIC*100 as procent_odrzuconych
FROM wnioski
GROUP BY jezyk
HAVING count(1)>500
ORDER BY 3 DESC;

--w podziale na kod przyjazdu

SELECT sz.kod_przyjazdu,count(1), count(
   case WHEN w.stan_wniosku LIKE 'odrz%' then 1 end)/
 count(1)::NUMERIC*100 as procent_odrzuconych
FROM wnioski w
join podroze p on w.id=p.id_wniosku
join szczegoly_podrozy sz on p.id=sz.id_podrozy
GROUP BY 1
HAVING count(1)>500
ORDER BY 3 DESC;

--wyjazdu

SELECT sz.kod_wyjazdu,count(1), count(
   case WHEN w.stan_wniosku LIKE 'odrz%' then 1 end)/
 count(1)::NUMERIC*100 as procent_odrzuconych
FROM wnioski w
join podroze p on w.id=p.id_wniosku
join szczegoly_podrozy sz on p.id=sz.id_podrozy
GROUP BY 1
HAVING count(1)>500
ORDER BY 3 DESC;