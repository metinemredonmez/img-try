'use client'

import Link from 'next/link'
import { ArrowRight, Video, Users, Briefcase, Sparkles } from 'lucide-react'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      {/* Header */}
      <header className="container mx-auto px-4 py-6">
        <nav className="flex items-center justify-between">
          <div className="text-2xl font-bold text-white">
            Vid<span className="text-blue-500">CV</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/auth/login" className="text-gray-300 hover:text-white transition">
              Giriş Yap
            </Link>
            <Link
              href="/auth/register"
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition"
            >
              Ücretsiz Başla
            </Link>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 text-center">
        <div className="inline-flex items-center gap-2 bg-blue-600/20 text-blue-400 px-4 py-2 rounded-full mb-6">
          <Sparkles className="w-4 h-4" />
          <span className="text-sm">AI Destekli Video CV Platformu</span>
        </div>

        <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
          CV'nizi <span className="text-blue-500">Konuşan Videoya</span>
          <br />Dönüştürün
        </h1>

        <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-10">
          Yapay zeka ile CV'nizden profesyonel video avatar oluşturun.
          İşverenlerin dikkatini çekin ve öne çıkın.
        </p>

        <div className="flex items-center justify-center gap-4">
          <Link
            href="/auth/register"
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-semibold flex items-center gap-2 transition"
          >
            Ücretsiz Dene <ArrowRight className="w-5 h-5" />
          </Link>
          <Link
            href="/demo"
            className="border border-gray-600 hover:border-gray-500 text-white px-8 py-4 rounded-lg font-semibold transition"
          >
            Demo İzle
          </Link>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-20">
        <h2 className="text-3xl font-bold text-white text-center mb-12">
          Neden VidCV?
        </h2>

        <div className="grid md:grid-cols-3 gap-8">
          <FeatureCard
            icon={<Video className="w-8 h-8 text-blue-500" />}
            title="AI Video Avatar"
            description="CV'nizden otomatik olarak konuşan profesyonel video oluşturun."
          />
          <FeatureCard
            icon={<Users className="w-8 h-8 text-green-500" />}
            title="Anonim Mod"
            description="Gizliliğinizi koruyun. AI-üretilmiş anonim avatar ile başvurun."
          />
          <FeatureCard
            icon={<Briefcase className="w-8 h-8 text-purple-500" />}
            title="AI Eşleştirme"
            description="İşverenlerin aradığı yeteneklerle otomatik eşleşin."
          />
        </div>
      </section>

      {/* Stats Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="bg-slate-800/50 rounded-2xl p-12 grid md:grid-cols-4 gap-8 text-center">
          <StatItem value="10,000+" label="Aktif Kullanıcı" />
          <StatItem value="5,000+" label="Video CV" />
          <StatItem value="500+" label="İşveren" />
          <StatItem value="30+" label="Desteklenen Dil" />
        </div>
      </section>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-12 border-t border-slate-700">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-gray-400">
            © 2024 VidCV - Deep Room AI. Tüm hakları saklıdır.
          </div>
          <div className="flex items-center gap-6 text-gray-400">
            <Link href="/privacy" className="hover:text-white transition">Gizlilik</Link>
            <Link href="/terms" className="hover:text-white transition">Kullanım Şartları</Link>
            <Link href="/contact" className="hover:text-white transition">İletişim</Link>
          </div>
        </div>
      </footer>
    </main>
  )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="bg-slate-800/50 p-8 rounded-xl hover:bg-slate-800 transition">
      <div className="mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </div>
  )
}

function StatItem({ value, label }: { value: string, label: string }) {
  return (
    <div>
      <div className="text-4xl font-bold text-white mb-2">{value}</div>
      <div className="text-gray-400">{label}</div>
    </div>
  )
}
