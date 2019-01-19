
SELECT partner, count(1),count(
   case WHEN stan_wniosku LIKE 'odrz%' then id end)/
 count(1)::NUMERIC as procent
FROM wnioski
GROUP BY partner
ORDER BY 2 DESC ;


SELECT kod_kraju,count(1), count(
   case WHEN stan_wniosku LIKE 'odrz%' then id end)/
 count(1)::NUMERIC as procent
FROM wnioski
GROUP BY kod_kraju
HAVING count(1)>1000
ORDER BY 3 DESC ;




select partner,count(1)
from wnioski
group by partner;

select count (distinct id)
from klienci;

select count(distinct id)
from wnioski;

select count(distinct email),count(distinct id)
from klienci;






