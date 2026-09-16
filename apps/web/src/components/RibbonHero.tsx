'use client';

/**
 * The hero: a tangled ribbon that separates into coloured strands (ADR-0011).
 *
 * The product's function, shown rather than described. One `InstancedMesh`
 * drives every segment, so the whole scene is one draw call regardless of
 * density.
 *
 * ADR-0011's performance rules are not suggestions, and each is here because
 * breaking it has a specific consequence:
 *   - mutate matrices inside useFrame, never setState per frame — a state
 *     update per frame re-renders React sixty times a second and the page
 *     stops responding to input;
 *   - cap dpr at 1.5 — a 3x retina display otherwise renders nine times the
 *     pixels for no visible gain;
 *   - pause off-screen and on visibilitychange — a backgrounded tab burning
 *     GPU is why laptops get hot on news sites;
 *   - watch the frame rate and degrade, then fall back to the poster.
 *
 * The poster path is a first-class experience, not a consolation: it is what
 * reduced-motion users, WebGL-less browsers and weak devices actually see.
 */

import { Canvas, useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';

const SEGMENTS = 220;
const STRANDS = 4;

/** Read once, at mount: these do not change without a reload. */
function useEnvironment() {
  const [env, setEnv] = useState({ reducedMotion: false, webgl: true, saveData: false });

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let webgl = false;
    try {
      const canvas = document.createElement('canvas');
      webgl = Boolean(canvas.getContext('webgl2'));
    } catch {
      webgl = false;
    }

    // Save-Data and a slow effective connection both mean "do not spend the
    // user's battery or bandwidth on decoration".
    const connection = (
      navigator as Navigator & {
        connection?: { saveData?: boolean; effectiveType?: string };
      }
    ).connection;
    const saveData =
      connection?.saveData === true ||
      (connection?.effectiveType !== undefined && /2g/.test(connection.effectiveType));

    setEnv({ reducedMotion: reduced, webgl, saveData });
  }, []);

  return env;
}

const STRAND_COLOURS = ['#4da3ff', '#3ecf9a', '#f5b942', '#e878c8'];

function Ribbons({ separation }: { separation: React.RefObject<number> }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Per-instance colour, uploaded once. Recomputing it per frame would be the
  // single most expensive thing in this scene.
  const colours = useMemo(() => {
    const array = new Float32Array(SEGMENTS * STRANDS * 3);
    const colour = new THREE.Color();
    for (let s = 0; s < STRANDS; s += 1) {
      colour.set(STRAND_COLOURS[s] ?? '#ffffff');
      for (let i = 0; i < SEGMENTS; i += 1) {
        const at = (s * SEGMENTS + i) * 3;
        array[at] = colour.r;
        array[at + 1] = colour.g;
        array[at + 2] = colour.b;
      }
    }
    return array;
  }, []);

  useFrame(({ clock }) => {
    const node = mesh.current;
    if (!node) return;

    const t = clock.getElapsedTime();
    const sep = separation.current;

    let index = 0;
    for (let s = 0; s < STRANDS; s += 1) {
      // At separation 0 every strand shares one path — the tangle. As it rises
      // they pull apart onto their own lanes, which is the whole metaphor.
      const lane = (s - (STRANDS - 1) / 2) * 1.5 * sep;
      const phase = s * 1.7;

      for (let i = 0; i < SEGMENTS; i += 1) {
        const u = i / SEGMENTS;
        const x = (u - 0.5) * 14;

        const tangled =
          Math.sin(u * 9 + t * 0.6 + phase) * 0.9 + Math.sin(u * 21 + t * 0.9 + phase) * 0.35;
        const separated = Math.sin(u * 11 + t * 1.1 + phase) * 0.28;

        dummy.position.set(
          x,
          lane + tangled * (1 - sep) + separated * sep,
          Math.cos(u * 7 + phase) * (1 - sep) * 1.2,
        );
        dummy.scale.set(0.055, 0.055 + Math.abs(separated) * 0.1 * sep, 0.055);
        dummy.updateMatrix();
        node.setMatrixAt(index, dummy.matrix);
        index += 1;
      }
    }
    node.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, SEGMENTS * STRANDS]}>
      <sphereGeometry args={[1, 6, 6]}>
        <instancedBufferAttribute attach="attributes-color" args={[colours, 3]} />
      </sphereGeometry>
      <meshBasicMaterial vertexColors toneMapped={false} />
    </instancedMesh>
  );
}

/** Drops to the poster after a sustained bad frame rate (ADR-0011). */
function FpsWatchdog({ onSlow }: { onSlow: () => void }) {
  const frames = useRef(0);
  const since = useRef(performance.now());
  const badWindows = useRef(0);

  useFrame(() => {
    frames.current += 1;
    const now = performance.now();
    if (now - since.current < 1000) return;

    const fps = (frames.current * 1000) / (now - since.current);
    frames.current = 0;
    since.current = now;

    // Two consecutive bad seconds, not one: a single slow second happens during
    // any page load and is not a reason to give up on the scene.
    badWindows.current = fps < 25 ? badWindows.current + 1 : 0;
    if (badWindows.current >= 2) onSlow();
  });

  return null;
}

function Poster() {
  return (
    <div
      aria-hidden
      className="absolute inset-0"
      style={{
        background:
          'radial-gradient(70% 55% at 30% 35%, var(--mesh-1), transparent 70%),' +
          'radial-gradient(60% 50% at 70% 60%, var(--mesh-2), transparent 70%),' +
          'radial-gradient(50% 40% at 50% 80%, var(--mesh-3), transparent 70%)',
      }}
    />
  );
}

export function RibbonHero() {
  const { reducedMotion, webgl, saveData } = useEnvironment();
  const [degraded, setDegraded] = useState(false);
  const [visible, setVisible] = useState(true);
  const separation = useRef(0);
  const container = useRef<HTMLDivElement>(null);

  // Scroll-linked, not time-based: the separation is something the visitor
  // performs rather than watches.
  useEffect(() => {
    function onScroll(): void {
      const top = container.current?.getBoundingClientRect().top ?? 0;
      const progress = Math.min(1, Math.max(0, -top / Math.max(1, window.innerHeight * 0.8)));
      separation.current = progress;
    }
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
    };
  }, []);

  // Stop rendering when off-screen or in a hidden tab.
  useEffect(() => {
    const node = container.current;
    if (!node) return;
    const observer = new IntersectionObserver(([entry]) => {
      setVisible(entry?.isIntersecting ?? true);
    });
    observer.observe(node);

    const onVisibility = (): void => {
      setVisible(document.visibilityState === 'visible');
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      observer.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  // WebGL must never initialise under reduced motion — not "initialise and sit
  // still". That is the difference between honouring the preference and
  // pretending to.
  const useCanvas = webgl && !reducedMotion && !saveData && !degraded;

  return (
    <div
      ref={container}
      className="relative h-[52vh] min-h-72 w-full overflow-hidden rounded-[var(--radius-lg)]"
    >
      {useCanvas ? (
        <Canvas
          dpr={[1, 1.5]}
          frameloop={visible ? 'always' : 'never'}
          camera={{ position: [0, 0, 9], fov: 50 }}
          gl={{ antialias: false, powerPreference: 'high-performance' }}
        >
          <Ribbons separation={separation} />
          <FpsWatchdog
            onSlow={() => {
              setDegraded(true);
            }}
          />
        </Canvas>
      ) : (
        <Poster />
      )}
    </div>
  );
}
