-- Sessions
create table vision_sessions (
  id              uuid primary key default gen_random_uuid(),
  api_key_hash    text not null,
  vehicle_context jsonb,
  status          text not null default 'open',
  created_at      timestamptz not null default now(),
  expires_at      timestamptz not null
);

-- Analysis calls audit log (created before session_images to avoid circular FK)
create table vision_analysis_calls (
  id              uuid primary key default gen_random_uuid(),
  call_type       text not null,
  session_id      uuid references vision_sessions(id),
  image_id        uuid,
  model           text not null,
  prompt_tokens   int,
  response_tokens int,
  latency_ms      int,
  status          text not null,
  error           text,
  raw_response    jsonb,
  created_at      timestamptz not null default now()
);

-- Session images
create table vision_session_images (
  id               uuid primary key default gen_random_uuid(),
  session_id       uuid not null references vision_sessions(id),
  image_url        text not null,
  angle            text,
  image_width      int,
  image_height     int,
  status           text not null default 'pending',
  damages          jsonb,
  gemini_call_id   uuid,
  error            text,
  verified_damages jsonb,
  verified_at      timestamptz,
  verified_by      text,
  uploaded_at      timestamptz not null default now(),
  analyzed_at      timestamptz
);

-- Consolidated damage maps
create table vision_damage_maps (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid not null references vision_sessions(id) unique,
  images      jsonb not null,
  zones       jsonb not null,
  summary     jsonb not null,
  image_count int not null,
  built_at    timestamptz not null default now()
);
