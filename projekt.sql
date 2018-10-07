
SELECT partner, count(
   case WHEN stan_wniosku LIKE 'odrz%' then id end)/
 count(1)::NUMERIC
FROM wnioski
GROUP BY partner
ORDER BY 2 DESC ;


select partner,count(1)
from wnioski
group by partner;







