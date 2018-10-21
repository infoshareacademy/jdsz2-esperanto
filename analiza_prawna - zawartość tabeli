-- ESP8_analiza_prawna - zbadac tabele:

-- analiza prawna zawiera tylko 77 wnioskow, z czego w kazdy dla innego nr referencyjnego
-- status jest niejasny, tylko 4 odrzucone
-- Wydaje się jedyną wartą poddania dalszej analizie kolumna agent_id (168)

Select count(1) from analiza_prawna;

Select count(a.id_wniosku), w.nr_referencyjny, a.status
from wnioski w
inner join analiza_prawna a on w.id = a.id_wniosku
group by w.nr_referencyjny, a.status
order by a.status asc;

Select a.agent_id, count(a.id_wniosku)
from wnioski w
inner join analiza_prawna a on w.id = a.id_wniosku
group by a.agent_id
order by 2 desc;
