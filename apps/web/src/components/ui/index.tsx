/**
 * Component primitives (docs/14-design-system.md).
 *
 * Every colour here is a token. Nothing takes a raw hex or an arbitrary
 * Tailwind palette value, because the contrast gate only checks the token file
 * — a stray `text-blue-400` is invisible to it and would ship unverified.
 *
 * These live on both surfaces (docs/28 §D1). The expressive props — `glow`,
 * `glass` — are opt-in and belong on landing, auth, upload and waiting. Inside
 * the player they are a bug: there the user is judging audio quality, and
 * decoration competes with the signal.
 */
'use client';

import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  TextareaHTMLAttributes,
} from 'react';
import { useId } from 'react';

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ');
}

// ------------------------------------------------------------------ Button --

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
type ButtonSize = 'sm' | 'md' | 'lg';

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-[oklch(0.15_0.01_265)] hover:bg-accent-hover border-transparent',
  secondary:
    'bg-bg-surface text-fg-primary border-border-strong hover:border-accent hover:text-accent',
  ghost:
    'bg-transparent text-fg-secondary border-transparent hover:text-fg-primary hover:bg-bg-surface',
  danger: 'bg-danger text-[oklch(0.98_0_0)] border-transparent hover:opacity-90',
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: 'text-sm px-3 py-1.5 gap-1.5',
  md: 'text-[0.95rem] px-4 py-2.5 gap-2',
  lg: 'text-base px-6 py-3 gap-2.5',
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Shows a spinner and blocks interaction without changing the button's width. */
  loading?: boolean;
  /** Expressive surfaces only. */
  glow?: boolean;
  full?: boolean;
}

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  glow = false,
  full = false,
  disabled,
  children,
  className,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled ?? loading}
      // Tells assistive tech the control is working, which a spinner alone does not.
      aria-busy={loading || undefined}
      className={cx(
        'inline-flex items-center justify-center rounded-[var(--radius-sm)] border font-medium',
        'transition-[background,border-color,color,box-shadow] duration-(--dur-fast) ease-(--ease-out)',
        'disabled:opacity-55 disabled:cursor-not-allowed',
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        full && 'w-full',
        glow && 'shadow-[var(--glow-accent)]',
        className,
      )}
    >
      {loading && <Spinner size={size === 'lg' ? 18 : 14} />}
      {children}
    </button>
  );
}

// ----------------------------------------------------------------- Spinner --

export function Spinner({ size = 16, label }: { size?: number; label?: string }) {
  return (
    <span
      role={label ? 'status' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      // Reduced motion: the ring stops rotating but stays visible, so the
      // "busy" meaning survives (NFR-A11Y-04).
      className="inline-block shrink-0 animate-spin rounded-full border-2 border-current border-t-transparent motion-reduce:animate-none"
      style={{ width: size, height: size }}
    />
  );
}

// -------------------------------------------------------------------- Card --

export function Card({
  children,
  className,
  glass = false,
  as: Tag = 'div',
}: {
  children: ReactNode;
  className?: string;
  /** Expressive surfaces only: translucent over a mesh background. */
  glass?: boolean;
  as?: 'div' | 'section' | 'article' | 'form';
}) {
  return (
    <Tag
      className={cx(
        'rounded-[var(--radius-lg)] border p-5',
        glass
          ? 'bg-[var(--glass-bg)] border-[var(--glass-border)] backdrop-blur-[var(--glass-blur)]'
          : 'bg-bg-surface border-border-subtle',
        className,
      )}
    >
      {children}
    </Tag>
  );
}

// ------------------------------------------------------------------- Field --

export interface FieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  hint?: string;
  error?: string;
}

/**
 * Label, control, hint and error wired together by id.
 *
 * `aria-describedby` and `aria-invalid` are set here rather than at each call
 * site: a form built from these cannot accidentally ship an error message that
 * a screen reader never announces.
 */
export function Field({ label, hint, error, id, className, ...rest }: FieldProps) {
  const auto = useId();
  const fieldId = id ?? auto;
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;

  return (
    <div className={cx('flex flex-col gap-1.5', className)}>
      <label htmlFor={fieldId} className="text-sm text-fg-secondary">
        {label}
      </label>
      <input
        {...rest}
        id={fieldId}
        aria-invalid={error ? true : undefined}
        aria-describedby={cx(hint && hintId, error && errorId) || undefined}
        className={cx(
          'w-full rounded-[var(--radius-sm)] border bg-bg-inset px-3 py-2.5 text-fg-primary',
          'placeholder:text-fg-muted',
          'transition-colors duration-(--dur-fast)',
          error ? 'border-danger' : 'border-border-strong focus:border-accent',
        )}
      />
      {hint && !error && (
        <p id={hintId} className="text-xs text-fg-muted">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} className="text-xs text-danger">
          {error}
        </p>
      )}
    </div>
  );
}

export interface TextAreaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label: string;
  hint?: string;
}

export function TextArea({ label, hint, id, className, ...rest }: TextAreaProps) {
  const auto = useId();
  const fieldId = id ?? auto;
  return (
    <div className={cx('flex flex-col gap-1.5', className)}>
      <label htmlFor={fieldId} className="text-sm text-fg-secondary">
        {label}
      </label>
      <textarea
        {...rest}
        id={fieldId}
        className="w-full rounded-[var(--radius-sm)] border border-border-strong bg-bg-inset px-3 py-2.5 text-fg-primary focus:border-accent"
      />
      {hint && <p className="text-xs text-fg-muted">{hint}</p>}
    </div>
  );
}

// ---------------------------------------------------------------- Progress --

export function Progress({
  value,
  max = 100,
  label,
  tone = 'accent',
}: {
  /** Omit for indeterminate — genuinely unknown progress, never a fake crawl. */
  value?: number;
  max?: number;
  label?: string;
  tone?: 'accent' | 'success' | 'warning';
}) {
  // Narrowed to a number here, so the determinate branch never has to assert
  // that it is one.
  const pct = typeof value === 'number' ? Math.min(100, Math.max(0, (value / max) * 100)) : null;
  const bg = { accent: 'bg-accent', success: 'bg-success', warning: 'bg-warning' }[tone];

  return (
    <div
      role="progressbar"
      aria-valuenow={pct === null ? undefined : Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
      className="h-2 w-full overflow-hidden rounded-full border border-border-subtle bg-bg-inset"
    >
      {pct === null ? (
        // Indeterminate is honest about not knowing. A bar that creeps to 99%
        // and stops is worse than one that admits it cannot estimate.
        <div className={cx('h-full w-1/3 animate-pulse motion-reduce:animate-none', bg)} />
      ) : (
        <div
          className={cx('h-full transition-[width] duration-(--dur-normal) ease-(--ease-out)', bg)}
          style={{ width: `${pct.toFixed(2)}%` }}
        />
      )}
    </div>
  );
}

// ------------------------------------------------------------------- Badge --

type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger';

const TONES: Record<Tone, string> = {
  neutral: 'text-fg-muted border-border-subtle',
  accent: 'text-accent border-accent',
  success: 'text-success border-success',
  warning: 'text-warning border-warning',
  danger: 'text-danger border-danger',
};

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: Tone }) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs',
        TONES[tone],
      )}
    >
      {children}
    </span>
  );
}

// ------------------------------------------------------------------- Alert --

export function Alert({
  tone = 'danger',
  title,
  children,
}: {
  tone?: Tone;
  title?: string;
  children: ReactNode;
}) {
  return (
    <div
      // assertive for errors, polite otherwise: an error the user must act on
      // should interrupt, a status update should not.
      role={tone === 'danger' ? 'alert' : 'status'}
      className={cx(
        'rounded-[var(--radius-md)] border px-4 py-3 text-sm',
        TONES[tone],
        'bg-bg-surface',
      )}
    >
      {title && <p className="mb-1 font-medium">{title}</p>}
      <div className="text-fg-secondary">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------- Skeleton --

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cx(
        'animate-pulse rounded-[var(--radius-sm)] bg-bg-elevated motion-reduce:animate-none',
        className,
      )}
    />
  );
}

// -------------------------------------------------------------- SpeakerDot --

/**
 * A speaker's identity colour.
 *
 * Always paired with its label by the caller: colour is never the sole
 * identifier (NFR-A11Y-06), which also matters at four speakers where magenta
 * and azure converge under deuteranopia.
 */
export function SpeakerDot({ index, size = 10 }: { index: number; size?: number }) {
  // A tuple rather than a plain array: indexing an array yields
  // `string | undefined` under noUncheckedIndexedAccess, and a speaker colour
  // that can be undefined is a colour that can silently vanish.
  const tokens = ['--spk-1', '--spk-2', '--spk-3', '--spk-4'] as const;
  const token = tokens[index % tokens.length] ?? '--spk-mixed';
  return (
    <span
      aria-hidden
      className="inline-block shrink-0 rounded-full"
      style={{ width: size, height: size, background: `var(${token})` }}
    />
  );
}
