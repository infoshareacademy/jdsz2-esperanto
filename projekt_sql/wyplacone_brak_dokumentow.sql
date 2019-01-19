--ESP-21

-- dwa z wnioskow odrzuconych przez operatora, do ktorych nie zostaly zalaczone
--ani dowod osobisty ani skan bilety zostaly wyplacone przez Airhelp
--WYLUDZENIE ? obydwa wnioski z kwietnia 2015 roku, być może zostało ty wyłapane na początku.. ?
--ZALECENIE dla AirHelp : Identyfikator operatora nie jest używany w tabelach, nie mozna polaczyć po nim
--wnioskow, również nazwy się nie pokrywają !
-- Dane niejednoznaczne z uwagi na zamazanie bazy danych do analizy..


Select
  count(w.id_wniosku),
  t.id,
  w.data_utworzenia,
  t.kwota_rekompensaty,
  sz.identyfikator_operator_operujacego,
  t.stan_wniosku,
  w.status as odp_operatora,
  w.id_agenta
from analizy_wnioskow w
join wnioski t on t.id = w.id_wniosku
join podroze o on o.id_wniosku = t.id
join szczegoly_podrozy sz on sz.id_podrozy = o.id
where w.skan_dowodu_ok is false
      and w.skan_biletu_ok is false
      and t.stan_wniosku like 'wypl%'
group by sz.identyfikator_operator_operujacego, t.id, w.status, w.data_utworzenia, w.id_agenta

Select * from wnioski
where id = 36710 or id = 36937;

Select * from analizy_wnioskow
where id_wniosku = 36710 or id_wniosku = 36937;

Select * from analiza_operatora
where id_wniosku = 36710 or id_wniosku = 36937;