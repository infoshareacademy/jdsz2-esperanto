-- ESP-9 - zbadac relacje wniosku z tabela analiza wniosku

-- w tabeli wnioski, 46,8 tys wnioskow przynalezy do 1 nr_referencyjnego (4BC8C532), co swiadczy o tym, ze nie jest to nr lotu,
-- w tabeli analiza_wnioskow jest 84,5 tyś rekordow, z czego 47,8 tyś dla agent_id 168
-- dla agenta

-- WNIOSKI: nr_referencyjny nie jest przydatny, ale dla AGENTA nr 168 99% wnioskow jest zaakcpetowana

select count (nr_referencyjny), nr_referencyjny from wnioski
group by nr_referencyjny
order by count (nr_referencyjny) desc;

'procent odrzuconych a agent id'

Select count(1) from analizy_wnioskow;

with analizy as
(Select
  id_agenta,
  count(id_agenta) as ilosc_analiz,
  count(case when status = 'odrzucony' then id_agenta end) as ilosc_odrzuconych,
  count(case when status = 'zaakceptowany' then id_agenta end) as ilosc_zaakceptowanych
from
  analizy_wnioskow
group by
  id_agenta
order by 2 desc)

Select id_agenta, ilosc_analiz,
  ilosc_odrzuconych / ilosc_analiz::numeric as proc_odrzuconych,
  ilosc_zaakceptowanych / ilosc_analiz::numeric as proc_zaakceptowanych
 from analizy
 where ilosc_analiz > 50



