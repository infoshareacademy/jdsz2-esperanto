--liczba wniosków złożona przez jednego klienta,
-- weryfikacja po mailu
-- wniosek bezużyeczne, zaszyfrowane dane
SELECT email, count(1)
FROM klienci
GROUP BY email
ORDER BY count(1) desc;