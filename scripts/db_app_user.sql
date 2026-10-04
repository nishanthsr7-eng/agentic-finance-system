-- Least-privilege MySQL user for the deployed API (run once in the TiDB SQL Editor
-- as the admin/root user). Replace <prefix> with the prefix of your TiDB user
-- names (the part before ".root") and <password> with a new strong password.
-- Never commit the real password.
--
-- Why more than SELECT/INSERT/UPDATE/DELETE: the API applies pending
-- migrations at startup (backend/migrate.py), which need CREATE (tables),
-- ALTER (add columns), INDEX and REFERENCES (foreign keys in CREATE TABLE).
-- Deliberately NOT granted: DROP, GRANT OPTION, and anything outside flux.*.

CREATE USER '<prefix>.flux_app'@'%' IDENTIFIED BY '<password>';

GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, REFERENCES
    ON flux.* TO '<prefix>.flux_app'@'%';

SHOW GRANTS FOR '<prefix>.flux_app'@'%';

-- Roll back (after pointing Render back at the admin user):
-- DROP USER '<prefix>.flux_app'@'%';
